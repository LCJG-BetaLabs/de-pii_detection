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

# MAGIC %pip install flair==0.13.1

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

from discoverx import DX

dx = DX()

# COMMAND ----------

dx.display_rules()


# COMMAND ----------

import logging
from typing import Optional, List, Tuple, Set

from presidio_analyzer import (
    RecognizerResult,
    EntityRecognizer,
    AnalysisExplanation,
)
from presidio_analyzer.nlp_engine import NlpArtifacts

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
except ImportError:
    print("Flair is not installed")


logger = logging.getLogger("presidio-analyzer")


class FlairRecognizer(EntityRecognizer):
    """
    Wrapper for a flair model, if needed to be used within Presidio Analyzer.

    :example:
    >from presidio_analyzer import AnalyzerEngine, RecognizerRegistry

    >flair_recognizer = FlairRecognizer()

    >registry = RecognizerRegistry()
    >registry.add_recognizer(flair_recognizer)

    >analyzer = AnalyzerEngine(registry=registry)

    >results = analyzer.analyze(
    >    "My name is Christopher and I live in Irbid.",
    >    language="en",
    >    return_decision_process=True,
    >)
    >for result in results:
    >    print(result)
    >    print(result.analysis_explanation)


    """

    ENTITIES = [
        "LOCATION",
        "PERSON",
        "ORGANIZATION",
        # "MISCELLANEOUS"   # - There are no direct correlation with Presidio entities.
    ]

    DEFAULT_EXPLANATION = "Identified as {} by Flair's Named Entity Recognition"

    CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"ORGANIZATION"}, {"ORG"}),
        # ({"MISCELLANEOUS"}, {"MISC"}), # Probably not PII
    ]

    MODEL_LANGUAGES = {
        "en": "flair/ner-english-large",
        # "es": "flair/ner-spanish-large",
        # "de": "flair/ner-german-large",
        # "nl": "flair/ner-dutch-large",
    }

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        # 'MISC': 'MISCELLANEOUS'   # - Probably not PII
    }

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        check_label_groups: Optional[Tuple[Set, Set]] = None,
        model: SequenceTagger = None,
    ):
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )

        supported_entities = supported_entities if supported_entities else self.ENTITIES
        self.model = (
            model
            if model
            else SequenceTagger.load(self.MODEL_LANGUAGES.get(supported_language))
        )

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="Flair Analytics",
        )

    def load(self) -> None:
        """Load the model, not used. Model is loaded during initialization."""
        pass

    def get_supported_entities(self) -> List[str]:
        """
        Return supported entities by this model.

        :return: List of the supported entities.
        """
        return self.supported_entities

    # Class to use Flair with Presidio as an external recognizer.
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        Analyze text using Text Analytics.

        :param text: The text for analysis.
        :param entities: Not working properly for this recognizer.
        :param nlp_artifacts: Not used by this recognizer.
        :param language: Text language. Supported languages in MODEL_LANGUAGES
        :return: The list of Presidio RecognizerResult constructed from the recognized
            Flair detections.
        """

        results = []

        sentences = Sentence(text)
        self.model.predict(sentences)

        # If there are no specific list of entities, we will look for all of it.
        if not entities:
            entities = self.supported_entities

        for entity in entities:
            if entity not in self.supported_entities:
                continue

            for ent in sentences.get_spans("ner"):
                if not self.__check_label(
                    entity, ent.labels[0].value, self.check_label_groups
                ):
                    continue
                textual_explanation = self.DEFAULT_EXPLANATION.format(
                    ent.labels[0].value
                )
                explanation = self.build_flair_explanation(
                    round(ent.score, 2), textual_explanation
                )
                flair_result = self._convert_to_recognizer_result(ent, explanation)

                results.append(flair_result)

        return results

    def _convert_to_recognizer_result(self, entity, explanation) -> RecognizerResult:

        entity_type = self.PRESIDIO_EQUIVALENCES.get(entity.tag, entity.tag)
        flair_score = round(entity.score, 2)

        flair_results = RecognizerResult(
            entity_type=entity_type,
            start=entity.start_position,
            end=entity.end_position,
            score=flair_score,
            analysis_explanation=explanation,
        )

        return flair_results

    def build_flair_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.

        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :return:
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        return any(
            [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
        )


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

flair_recognizer = (
        FlairRecognizer()
    )  # This would download a very large (+2GB) model on the first run

registry = RecognizerRegistry()
registry.add_recognizer(flair_recognizer)
# Define the analyzer, and add custom matchers if needed
analyzer = AnalyzerEngine(registry=registry)

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
    .filter('max_score > 0.5 AND score < 0.85 AND score > 0.1')
    .orderBy('table_name','column_name', 'entity_type')
)

# TODO: Comment out the display when saving the result to table
summarised_detections.display()

# COMMAND ----------

# TODO: Store result to a table
# summarised_detections.write.saveAsTable("default..")

# COMMAND ----------


