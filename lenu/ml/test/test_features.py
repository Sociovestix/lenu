import pandas  # type: ignore
import numpy

from lenu.data.elf_codes import ELFAbbreviations
from lenu.ml.features import ELFAbbreviationTransformer


class TestELFAbbreviationsTransformer:
    def test_elf_abbreviation_transformer(self):
        elf_abbr = ELFAbbreviations(
            pandas.DataFrame(
                [
                    {"Jurisdiction": "DE", "ELF Code": "2HBR", "Abbreviation": "GmbH"},
                    {"Jurisdiction": "DE", "ELF Code": "40DB", "Abbreviation": "OHG"},
                    {
                        "Jurisdiction": "DE",
                        "ELF Code": "40DB",
                        "Abbreviation": "OHG mbH",
                    },
                ]
            )
        )

        elf_abbr_transformer = ELFAbbreviationTransformer(elf_abbr)

        res = elf_abbr_transformer.transform(
            pandas.DataFrame(
                {
                    "Entity.LegalName": ["Hallo GmbH", "Hello OHG mbH"],
                    "Jurisdiction": ["DE", "DE"],
                }
            )
        )

        assert (res == numpy.array([[1, 0, 0], [0, 0, 1]])).all().all()

        assert list(res.columns) == ["abbr(GmbH)", "abbr(OHG)", "abbr(OHG mbH)"]
