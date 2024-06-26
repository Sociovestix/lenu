from functools import lru_cache

import numpy
import pandas  # type: ignore

ELF_CODE_FILE_NAME = "2023-09-28-elf-code-list-v1.5.csv"


def get_jurisdiction(elf_code_list):
    # We picked region code if we have one, otherwise Country Code
    return elf_code_list.apply(
        lambda x: x["Country sub-division code (ISO 3166-2)"]
        if pandas.notnull(x["Country sub-division code (ISO 3166-2)"])
        else x["Country Code (ISO 3166-1)"],
        axis=1,
    )


def expand_us_states(elf_code_list):
    # In US we have ELF Codes on federal level and ELF Codes on State level
    # We expand the list and for each state we add the federal level ELF codes
    # to that state's list of ELF Codes
    ex_us = elf_code_list[elf_code_list["Country Code (ISO 3166-1)"] != "US"]
    us_federal = elf_code_list[
        (elf_code_list["Country Code (ISO 3166-1)"] == "US")
        & (elf_code_list["Country sub-division code (ISO 3166-2)"].isnull())
    ]
    us_states = elf_code_list[
        (elf_code_list["Country Code (ISO 3166-1)"] == "US")
        & (elf_code_list["Country sub-division code (ISO 3166-2)"].notnull())
    ]

    def _assign_to_state(us_federal, state):
        us_federal_ = us_federal.copy()
        us_federal_["Country sub-division code (ISO 3166-2)"] = state
        return us_federal_

    # note: original federal level rows have been removed.
    return pandas.concat(
        [ex_us, us_states]
        + [
            _assign_to_state(us_federal, state)
            for state in us_states["Country sub-division code (ISO 3166-2)"].unique()
        ]
    )


class ELFAbbreviations:
    def __init__(self, elf_abbreviations_list):
        self._abbr_by_jurisdiction = elf_abbreviations_list.groupby("Jurisdiction")[
            "Abbreviation"
        ].apply(lambda abbreviations: list(numpy.unique(abbreviations)))
        self._elf_by_jur_and_abbr = elf_abbreviations_list.groupby(
            ["Jurisdiction", "Abbreviation"]
        )["ELF Code"].apply(lambda elf_codes: list(numpy.unique(elf_codes)))

    def abbreviations_for_jurisdiction(self, jurisdiction):
        return self._abbr_by_jurisdiction.get(jurisdiction, [])

    def elf_codes_for_abbreviation(self, jurisdiction, abbreviation):
        return self._elf_by_jur_and_abbr.get((jurisdiction, abbreviation), [])

    @staticmethod
    @lru_cache(maxsize=None)
    def matches(legal_name, abbr, use_lowercasing=True, use_endswith=True):
        if use_lowercasing:
            abbr = abbr.lower()
            legal_name = legal_name.lower()

        if use_endswith:
            if legal_name.endswith(" " + abbr):
                return True
        else:
            if abbr in legal_name:
                return True
        return False


class ELFCodeList:
    def __init__(self, elf_code_list):
        self.elf_code_list = elf_code_list

    def get_names(self):
        return self.elf_code_list.groupby("ELF Code").first()[
            ["Entity Legal Form name Local name"]
        ]

    def get_abbreviations(self) -> ELFAbbreviations:
        elf_code_list = expand_us_states(self.elf_code_list)

        elf_abbreviations = elf_code_list.assign(
            Jurisdiction=get_jurisdiction(elf_code_list)
        )[["Jurisdiction", "ELF Code", "Abbreviations Local language"]].dropna()

        elf_abbreviations = (
            elf_abbreviations.assign(
                Abbreviation=lambda d: d["Abbreviations Local language"].str.split(";")
            )
            .explode("Abbreviation")
            .drop_duplicates(["Jurisdiction", "ELF Code", "Abbreviation"])
        )

        return ELFAbbreviations(elf_abbreviations)

    def get_inactive_elf_codes(self):
        return list(self.elf_code_list[
            self.elf_code_list['ELF Status ACTV/INAC'] == 'INAC'
        ]['ELF Code'].unique())


def load_elf_code_list(url) -> ELFCodeList:
    elf_code_list = pandas.read_csv(
        url,
        low_memory=False,
        dtype=str,
        # the following will prevent pandas from converting strings like 'NA' to NaN.
        na_values=[""],
        keep_default_na=False,
    )
    return ELFCodeList(elf_code_list)
