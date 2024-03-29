import logging
from pathlib import Path

import joblib  # type: ignore
import numpy
import pandas  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.metrics import accuracy_score, balanced_accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.naive_bayes import ComplementNB  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from lenu.data import DataRepo, ELFAbbreviations, ELFCodeList
from lenu.data.lei import COL_LEGALNAME, COL_ELF
from lenu.ml.cnames import tokenize
from lenu.ml.features import ELFAbbreviationTransformer

logger = logging.getLogger(__name__)


def DefaultPipeline(elf_abbreviations: ELFAbbreviations, jurisdiction: str):
    feature_extractor = ColumnTransformer(
        transformers=[
            (
                "abbreviations",
                ELFAbbreviationTransformer(
                    elf_abbreviations=elf_abbreviations,
                    jurisdiction=jurisdiction,
                ),
                0,  # column nr
            ),
            (
                "tokenizer",
                CountVectorizer(tokenizer=tokenize, lowercase=False, binary=True),
                0,  # column nr
            ),
        ]
    )
    pipeline_extPrep = Pipeline(
        steps=[
            ("feature_extraction", feature_extractor),
            ("classifier", ComplementNB()),
        ]
    )
    return pipeline_extPrep


def filter_infrequent_elf_codes(jurisdiction_data):
    # This fixes:
    # "ValueError: The least populated class in y has only 1 member, which is too few."
    filtered = jurisdiction_data.groupby(COL_ELF).filter(lambda x: len(x) >= 2)

    removed = jurisdiction_data[~jurisdiction_data[COL_ELF].isin(filtered[COL_ELF])]
    if len(removed) > 0:
        removed_elfs = list(removed[COL_ELF].unique())
        logger.warning(
            f"ELF Codes have been removed that appear only once in the training data: {removed_elfs}"
        )

    return filtered


def filter_inactive_elf_codes(jurisdiction_data, elf_code_list: ELFCodeList):
    return jurisdiction_data[
        ~jurisdiction_data[COL_ELF].isin(elf_code_list.get_inactive_elf_codes())
    ]


def train_for_jurisdiction(jurisdiction_data, pipeline, test_size=1.0 / 3):

    X = jurisdiction_data[[COL_LEGALNAME]].values
    y = jurisdiction_data[COL_ELF].values

    # The minimum number of groups for any class cannot be less than 2.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    logger.info(f"Model Accuracy: {accuracy}")
    bal_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    logger.info(f"Model Balanced Accuracy: {bal_accuracy}")

    return pipeline


class ELFDetectionModel:
    def __init__(self, jurisdiction, pipeline):
        self.jurisdiction = jurisdiction
        self.pipeline = pipeline

    def detect(self, legal_name, top=3):
        # preparing the input so that it fits
        input = numpy.array([[legal_name]])

        # do the prediction
        elf_probabilities = (
            pandas.Series(
                self.pipeline.predict_proba(input)[0], index=self.pipeline.classes_
            )
            .sort_values(ascending=False)
            .head(top)
        )

        return elf_probabilities


class ModelRepo:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def train_pipeline(self, jurisdiction, data_loader: DataRepo):
        jurisdiction_data = data_loader.load_lei_cdf_data(jurisdiction)
        elf_code_list = data_loader.load_elf_code_list()

        jurisdiction_data = filter_infrequent_elf_codes(jurisdiction_data)
        jurisdiction_data = filter_inactive_elf_codes(jurisdiction_data, elf_code_list)

        pipeline = DefaultPipeline(elf_code_list.get_abbreviations(), jurisdiction)

        nsamples = len(jurisdiction_data)
        logger.info(
            f"Train model for jurisdiction {jurisdiction} ({nsamples} samples) ..."
        )
        pipeline = train_for_jurisdiction(jurisdiction_data, pipeline)

        model_file = self.models_dir.joinpath(f"complement_nb_{jurisdiction}.joblib")
        logger.info(f"Store model to {self.models_dir} ...")
        joblib.dump(pipeline, model_file)

    def get_model(self, jurisdiction) -> ELFDetectionModel:
        model_file = self.models_dir.joinpath(f"complement_nb_{jurisdiction}.joblib")

        if not model_file.exists():
            raise ValueError(
                f"No model for Jurisdiction {jurisdiction} in {self.models_dir}"
            )

        pipeline = joblib.load(model_file)

        return ELFDetectionModel(jurisdiction, pipeline)

    def list(self):
        return list(
            sorted(
                [
                    model_path.stem.split("_")[-1]
                    for model_path in self.models_dir.glob("*.joblib")
                ]
            )
        )

    @staticmethod
    def from_models_dir(models_dir: Path) -> "ModelRepo":
        if not models_dir.exists() and not models_dir.is_dir():
            raise ValueError("Given model_dir does not exist or is not a directory")

        return ModelRepo(models_dir)
