from pathlib import Path
import os
import urllib
from typing import Optional
import shutil

from lenu import data
from lenu.data.elf_codes import load_elf_code_list, ELFAbbreviations, ELF_CODE_FILE_NAME
from lenu.data.lei import (
    load_lei_cdf_data,
    COL_LEGALNAME,
    COL_JURISDICTION,
    COL_ELF,
    get_legal_jurisdiction,
)
from lenu.data.goldencopyfiles import (
    GoldenCopyFilePublications,
    GoldenCopyFilePublication,
)

from logging import getLogger

try:
    from importlib import resources
except ImportError:
    # backport for python < 3.9
    import importlib_resources as resources


logger = getLogger(__name__)


class DataRepoNotReady(Exception):
    pass


class DataRepo:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def latest_lei_file(self) -> Optional[Path]:
        lei_files = list(
            sorted(self.data_dir.glob("*-gleif-goldencopy-lei2-golden-copy.csv.zip"))
        )
        latest_lei_file = lei_files[-1] if lei_files else None
        return latest_lei_file

    def elf_code_list_file(self) -> Optional[Path]:
        elf_file = Path(self.data_dir).joinpath(ELF_CODE_FILE_NAME)
        return elf_file if elf_file.exists() else None

    def ready(self) -> bool:
        return bool(self.latest_lei_file()) and bool(self.elf_code_list_file())

    def load_lei_cdf_data(self, jurisdiction):
        if not self.ready():
            raise DataRepoNotReady()

        logger.info(f"Loading LEI data into memory ({self.latest_lei_file()})")
        lei_data = load_lei_cdf_data(
            url=self.latest_lei_file(),
            usecols=[
                "LEI",
                COL_LEGALNAME,
                COL_JURISDICTION,
                COL_ELF,
                "Entity.LegalAddress.Region",
            ],
        ).assign(Jurisdiction=lambda d: d.apply(get_legal_jurisdiction, axis=1))
        jurisdiction_data = lei_data[lei_data["Jurisdiction"] == jurisdiction]
        return jurisdiction_data

    def load_elf_code_list(self):
        if not self.ready():
            raise DataRepoNotReady()

        logger.info(f"Loading ELF Code list ({self.elf_code_list_file()})")
        return load_elf_code_list(self.elf_code_list_file())

    def load_elf_abbreviations(self) -> ELFAbbreviations:
        elf_code_list = self.load_elf_code_list()
        elf_abbreviations = ELFAbbreviations.from_elf_code_list(elf_code_list)
        return elf_abbreviations

    def download_latest(self) -> None:
        publications = GoldenCopyFilePublications()
        publication: GoldenCopyFilePublication = publications.fetch_latest()

        lei_data_url = publication.lei2.full_file.csv.url
        filename = os.path.basename(lei_data_url)

        logger.info(f"Downloading {filename} to {self.data_dir}")
        urllib.request.urlretrieve(lei_data_url, self.data_dir.joinpath(filename))

        logger.info(f"Provide ELF Code list to {self.data_dir}")
        elf_target = self.data_dir.joinpath(ELF_CODE_FILE_NAME)
        with resources.path(data, ELF_CODE_FILE_NAME) as elf_resource:
            shutil.copy(elf_resource, elf_target)

    @staticmethod
    def from_data_dir(data_dir: Path) -> "DataRepo":
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(
                f"Given data_dir {str(data_dir)} does not exist or is not a directory."
            )
        return DataRepo(data_dir)
