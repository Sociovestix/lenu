import numpy
import pandas  # type: ignore
from lenu.data.elf_codes import ELFAbbreviations
from sklearn.base import TransformerMixin  # type: ignore


class ELFAbbreviationTransformer(TransformerMixin):
    def __init__(
        self,
        elf_abbreviations: ELFAbbreviations,
        jurisdiction: str,
        use_endswith=True,
        use_lowercasing=True,
    ):
        self.elf_abbreviations = elf_abbreviations
        self.jurisdiction = jurisdiction
        self.use_endswith = use_endswith
        self.use_lowercasing = use_lowercasing

    def fit(self, X, y=None):
        # does nothing, but needs to be implemented
        return self

    def transform(self, X, y=None):
        abbreviations = self.elf_abbreviations.abbreviations_for_jurisdiction(
            self.jurisdiction
        )

        predictions = []
        for name in X:
            pred = []  # list can end up empty
            for abbr in abbreviations:
                if self.elf_abbreviations.matches(
                    name, abbr, self.use_lowercasing, self.use_endswith
                ):
                    pred.append(1)
                else:
                    pred.append(0)

            predictions.append(pred)  # list can be empty
        return pandas.DataFrame(
            numpy.array(predictions),
            columns=[f"abbr({abbr})" for abbr in abbreviations],
        )

    def get_params(self, **kwargs):
        return {
            "elf_abbreviations": self.elf_abbreviations,
            "jurisdiction": self.jurisdiction,
            "use_endswith": self.use_endswith,
            "use_lowercasing": self.use_lowercasing,
        }
