from pathlib import Path
import sys
from logging import getLogger
from typing import Optional

import typer
from typer import Typer, echo

from lenu.data import DataRepo

from lenu.ml.pipelines import ModelRepo
from lenu.util import typer_log_config

logger = getLogger(__name__)

DEFAULT_DATA_DIR = "./data"
DEFAULT_MODEL_DIR = "./models"


app = Typer(callback=typer_log_config)


@app.command()
def download(
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, exists=True, dir_okay=True, resolve_path=True
    )
):
    """
    Download latest Golden Copy LEI file(s) and ELF Code list in csv format.
    :param data_dir: where to store the files
    """

    data_repo = DataRepo.from_data_dir(data_dir)
    data_repo.download_latest()


@app.command()
def train(
    jurisdiction: str,
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, exists=True, dir_okay=True, resolve_path=True
    ),
    models_dir: Path = typer.Option(
        DEFAULT_MODEL_DIR, exists=True, dir_okay=True, resolve_path=True
    ),
):
    data_repo = DataRepo.from_data_dir(data_dir)
    model_repo = ModelRepo.from_models_dir(models_dir)

    if data_repo.ready():
        model_repo.train_pipeline(jurisdiction, data_repo)
    else:
        logger.error("LEI data is not ready yet, Please use `lenu download`")


@app.command()
def elf(
    jurisdiction: str,
    legal_name: str,
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, exists=True, dir_okay=True, resolve_path=True
    ),
    models_dir: Path = typer.Option(
        DEFAULT_MODEL_DIR, exists=True, dir_okay=True, resolve_path=True
    ),
):
    data_repo = DataRepo.from_data_dir(data_dir)

    if not data_repo.ready():
        logger.error("LEI data is not ready yet, Please use `lenu download`")
        sys.exit(1)

    model_repo = ModelRepo.from_models_dir(models_dir)

    try:
        elf_model = model_repo.get_model(jurisdiction)
    except ValueError as ve:
        logger.error(ve.args[0])
        sys.exit(1)

    elf_probabilities = elf_model.detect(legal_name, top=3)

    # map things back to ELF Code and present
    elf_codes = data_repo.load_elf_code_list().groupby("ELF Code").first()
    res = (
        elf_codes.loc[elf_probabilities.index]
        .assign(Score=elf_probabilities)
        .reset_index()
        .rename(columns={"index": "ELF Code"})[
            ["ELF Code", "Entity Legal Form name Local name", "Score"]
        ]
    )

    echo(res)


@app.command()
def abbreviations(
    jurisdiction: str,
    abbr: Optional[str] = None,
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, exists=True, dir_okay=True, resolve_path=True
    ),
):
    data_repo = DataRepo.from_data_dir(data_dir)

    if not data_repo.ready():
        logger.error("LEI data is not ready yet, Please use `lenu download`")
        sys.exit(1)

    elf_abbr = data_repo.load_elf_abbreviations()
    if abbr:
        echo(f'ELF Codes for "{abbr}" in Jurisdiction {jurisdiction}:')
        echo(elf_abbr.elf_codes_for_abbreviation(jurisdiction, abbr))
    else:
        echo(f"Abbreviations in Jurisdiction {jurisdiction}:")
        echo(elf_abbr.abbreviations_for_jurisdiction(jurisdiction))


if __name__ == "__main__":
    app()
