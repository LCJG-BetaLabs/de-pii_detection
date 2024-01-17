# Databricks notebook source
# MAGIC %md
# MAGIC # PII detection with DiscoverX & Presidio
# MAGIC
# MAGIC This notebooks uses [DiscoverX](https://github.com/databrickslabs/discoverx) to run PII detection with [Presidio](https://microsoft.github.io/presidio/) over a set of tables in Unity Catalog.
# MAGIC
# MAGIC The notebook will:
# MAGIC 1. Use DiscoverX to sample a set of tables from Unity Catalog and unpivot all string columns into a long format dataset
# MAGIC 2. Run PII detection with Presidio
# MAGIC 3. Compute summarised statistics per table and column

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies

# COMMAND ----------

# MAGIC %pip install presidio_analyzer==2.2.33 dbl-discoverx==0.0.7

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download detection model

# COMMAND ----------

# MAGIC %sh python -m spacy download en_core_web_lg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define variables

# COMMAND ----------

# TODO: Change the table selection
#lc_dev.ml_house_of_fragrance_silver

# from_tables = "sample_data_discoverx.*.*"
from_tables = 'lc_prd.crm_db_neo_silver.*'

# TODO: Change the num of rows to sample
sample_size = 100


# COMMAND ----------

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry
from pyspark.sql.functions import pandas_udf, col, concat, lit, explode, count, avg, min, max, sum
from pyspark.sql.types import ArrayType, StringType, StructType, FloatType, StructField
from typing import Iterator

# COMMAND ----------

from discoverx.rules import RegexRule

customized_rules = [
    RegexRule(
    name="BRAND_NAME",
    description="LC Brand Name",
    definition=r"(\b[A-Z]+[A-Za-z\s]+\b)",
    match_example=["Bethan Gray","LAB SERIES","TATEOSSIAN", "PROTECTOR DAILY"],
    # nomatch_example=["71731 - Bethan Gray","69032 - LAB SERIES","10776 - TATEOSSIAN","11731 - PROTECTOR DAILY"],
    ),
    # RegexRule(
    # name="BRAND_CODE",
    # description="Brand Code",
    # definition=r"\b[A-Z]+\b(?:\s+[A-Z]+){1,2}\b",
    # match_example=["WW", "COS", "MW", "ABC DEF", "XYZ PQR STU"],
    # nomatch_example=["123", "ABC 123", "AB"],
    # ),
    RegexRule(
    name="CATEGORY_CODE",
    description="Category Code",
    definition=r"\b\d{3}\b",
    match_example=["123", "456", "789"],
    nomatch_example=["12", "4567", "abcd"],
    ),
#     RegexRule(
#     name="HK_ADDRESS",
#     description="Hong Kong Address",
#     definition=r"\b(?:[A-Za-z]+\s+){1,2}(?:Road|Street|Avenue|Lane|Drive|Court|Crescent|Plaza|Square|Place|Garden|Building)\b,\s*Hong\s*Kong\b",
#     match_example=["123 Main Street, Hong Kong", "1 Nathan Road, Hong Kong", "Block A, Times Square, Hong Kong"],
#     nomatch_example=["456 Washington Blvd, New York", "2 Oxford Street, London"],
# )
]


# COMMAND ----------

from discoverx import DX

dx = DX(custom_rules=customized_rules)

# COMMAND ----------

dx.display_rules()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform all sampled tables to long format
# MAGIC
# MAGIC This will take all columns of type string and unpivot (melt) them into a long format dataset

# COMMAND ----------

unpivoted_df = (
    dx.from_tables(from_tables)
    .unpivot_string_columns(sample_size=sample_size)
    .apply()
    .localCheckpoint()  # Checkpointing to reduce the query plan size
)

# unpivoted_df.display()

# COMMAND ----------

unpivoted_stats = unpivoted_df.groupBy("table_catalog", "table_schema", "table_name", "column_name").agg(
    count("string_value").alias("sampled_rows_count")
)

# unpivoted_stats.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Presidio UDFs
# MAGIC

# COMMAND ----------

from presidio_analyzer import PatternRecognizer, EntityRecognizer
person_recognizer = PatternRecognizer(supported_entity="PERSON",
                                      
                                      deny_list=["Bethan Gray","LAB SERIES","TATEOSSIAN", "PROTECTOR DAILY", "123", "456", "789"]
                                      )

entity_recognizer = EntityRecognizer(supported_entities='PERSON',
                                     supported_language='en',
                                     )


# COMMAND ----------

registry = RecognizerRegistry()
registry.load_predefined_recognizers()
registry.remove_recognizer('IbanRecognizer')
registry.remove_recognizer('AuAbnRecognizer')
registry.remove_recognizer('UsLicenseRecognizer')
registry.remove_recognizer('UsBankRecognizer')
registry.remove_recognizer('UsSsnRecognizer')
registry.remove_recognizer('AuTfnRecognizer')
registry.remove_recognizer('UsItinRecognizer')
registry.remove_recognizer('AuMedicareRecognizer')
registry.remove_recognizer('AuAcnRecognizer')
registry.remove_recognizer('UsPassportRecognizer')
registry.remove_recognizer('NhsRecognizer')
registry.remove_recognizer('SgFinRecognizer')

# Add the recognizer to the existing list of recognizers
registry.add_recognizer(person_recognizer)
registry.add_recognizer(entity_recognizer)
# Define the analyzer, and add custom matchers if needed
analyzer = AnalyzerEngine(default_score_threshold= 0.1, registry=registry)

# COMMAND ----------

analyzer.get_recognizers()

# COMMAND ----------



# broadcast the engines to the cluster nodes
broadcasted_analyzer = sc.broadcast(analyzer)


# define a pandas UDF function and a series function over it.
def analyze_text(text: str, analyzer: AnalyzerEngine) -> list[str]:
    try:
        analyzer_results = analyzer.analyze(text=text, language="en")
        dic = {}
        # Deduplicate the detections and take the max scode per entity type
        for r in analyzer_results:
            if r.entity_type in dic.keys():
                dic[r.entity_type] = max(r.score, dic[r.entity_type])
            else:
                dic[r.entity_type] = r.score
        return [{"entity_type": k, "score": dic[k]} for k in dic.keys()]
    except:
        return []


# define the iterator of series to minimize
def analyze_series(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    analyzer = broadcasted_analyzer.value
    for series in iterator:
        # Use that state for whole iterator.
        yield series.apply(lambda t: analyze_text(t, analyzer))


# define a the function as pandas UDF
analyze = pandas_udf(
    analyze_series,
    returnType=ArrayType(StructType([StructField("entity_type", StringType()), StructField("score", FloatType())])),
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Run PII detections

# COMMAND ----------

detections = (
    unpivoted_df.withColumn(
        "text", concat(col("table_name"), lit(" "), col("column_name"), lit(": "), col("string_value"))
    )
    .withColumn("detection", explode(analyze(col("text"))))
    .select("table_catalog", "table_schema", "table_name", "column_name", "string_value", "detection.*")
)

# detections.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute summarised statistics

# COMMAND ----------


summarised_detections = (
    detections.groupBy("table_catalog", "table_schema", "table_name", "column_name", "entity_type")
    .agg(count("string_value").alias("value_count"), max("score").alias("max_score"), sum("score").alias("sum_score"))
    .join(unpivoted_stats, ["table_catalog", "table_schema", "table_name", "column_name"])
    .withColumn("score", col("sum_score") / col("sampled_rows_count"))
    .select("table_catalog", "table_schema", "table_name", "column_name", "entity_type", "score", "max_score")
    .filter('max_score > 0.5 AND score < 0.8 AND score > 0.1')
    .orderBy('table_name','column_name', 'entity_type')
)

# TODO: Comment out the display when saving the result to table
summarised_detections.display()

# COMMAND ----------


summarised_detections = (
    detections.groupBy("table_catalog", "table_schema", "table_name", "column_name", "entity_type")
    .agg(count("string_value").alias("value_count"), max("score").alias("max_score"), sum("score").alias("sum_score"))
    .join(unpivoted_stats, ["table_catalog", "table_schema", "table_name", "column_name"])
    .withColumn("score", col("sum_score") / col("sampled_rows_count"))
    .select("table_catalog", "table_schema", "table_name", "column_name", "entity_type", "score", "max_score")
    .orderBy('table_name','column_name', 'entity_type')
)

# TODO: Comment out the display when saving the result to table
summarised_detections.display()

# COMMAND ----------

# TODO: Store result to a table
# summarised_detections.write.saveAsTable("default..")

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC select distinct brand_desc from lc_prd.crm_db_neo_silver.dbo_v_sales_dtl

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from lc_prd.crm_db_neo_silver.dbo_v_datamart_snapshot
# MAGIC limit 20
# MAGIC

# COMMAND ----------


