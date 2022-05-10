import pandas  # type: ignore
from lenu.data.elf_codes import ELFAbbreviations
from sklearn.base import (  # type: ignore
    BaseEstimator,
    ClassifierMixin,
)
from sklearn.utils.validation import check_is_fitted  # type: ignore


class ELFAbbreviationClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        elf_abbreviations: ELFAbbreviations,
        use_endswith=True,
        use_lowercasing=True,
    ):
        self.elf_abbreviations = elf_abbreviations
        self.use_endswith = use_endswith
        self.use_lowercasing = use_lowercasing

    def _get_jurisdiction_from_input(self, X):
        return X["Jurisdiction"].iloc[0]

    def fit(self, _, y):
        self.frequencies_ = pandas.value_counts(y)
        self.most_frequent_ = self.frequencies_.idxmax()
        return self

    def _best_elf_code_for_abbr(self, jurisdiction, name, abbr):
        if self.elf_abbreviations.matches(
            name, abbr, self.use_lowercasing, self.use_endswith
        ):
            elf_codes = self.elf_abbreviations.elf_codes_for_abbreviation(
                jurisdiction, abbr
            )
            pred = max(
                elf_codes, key=lambda elf_code: self.frequencies_.get(elf_code, 0)
            )
            return pred
        else:
            return None

    def predict(self, data):
        check_is_fitted(self, ["frequencies_", "most_frequent_"])

        jurisdiction = self._get_jurisdiction_from_input(data)

        abbreviations = self.elf_abbreviations.abbreviations_for_jurisdiction(
            jurisdiction
        )

        predictions = []
        for name in data["Entity.LegalName"]:
            potential_elf_codes = [
                self._best_elf_code_for_abbr(jurisdiction, name, abbr)
                for abbr in abbreviations
            ]

            pred = max(
                [elf_code for elf_code in potential_elf_codes if elf_code is not None],
                key=lambda elf_code: self.frequencies_.get(elf_code, 0),
                default=self.most_frequent_,
            )

            predictions.append(pred)
        return predictions
