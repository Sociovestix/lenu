from pathlib import Path
import sys
from logging import getLogger
from typing import Optional

import typer
from typer import Typer, echo

from lenu.data import DataRepo

from lenu.ml.pipelines import ModelRepo
from lenu.util import typer_log_config
from lenu.modelhub import (
    get_available_lenu_models_from_huggingface,
    get_model_from_huggingface,
)


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
    Download latest LEI data from gleif.org
    :param data_dir: where to store the files
    """

    data_repo = DataRepo.from_data_dir(data_dir)
    echo(
        "Downloading latest LEI data and ELF Codes for training from https://gleif.org ..."
    )
    echo(f"This may take a few minutes. Data will be downloaded to {str(data_dir)}")
    data_repo.download_latest()
    echo("Download finished.")


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
    """
    Train an ELF Detection model for a Jurisdiction.
    """
    data_repo = DataRepo.from_data_dir(data_dir)
    model_repo = ModelRepo.from_models_dir(models_dir)

    if data_repo.ready():
        echo(f"Training model for {jurisdiction} based on scikit-learn ...")
        echo(f"This may take a few minutes.")
        model_repo.train_pipeline(jurisdiction, data_repo)
        echo(f"Training finished. Model stored to {str(models_dir)}")
    else:
        logger.error("LEI data is not ready yet, Please use `lenu download`")


@app.command()
def list(
    models_dir: Path = typer.Option(
        DEFAULT_MODEL_DIR, exists=True, dir_okay=True, resolve_path=True
    )
):
    """
    List available ELF Detection models.
    """
    model_repo = ModelRepo.from_models_dir(models_dir)

    local_models = model_repo.list()
    if local_models:
        echo(
            "=== LENU ELF Detection models trained locally by applying scikit-learn approach ==="
        )
        for m in local_models:
            echo(m)
        echo("")

    remote_models = get_available_lenu_models_from_huggingface()
    if remote_models:
        echo(
            "=== LENU ELF Detection Transformer models available on https://huggingface.co/Sociovestix (recommended) ==="
        )
        for m in remote_models:
            echo(m)
        echo("")
    echo("You can detect ELF Codes for a jurisdiction with these models like this:")
    echo('lenu elf {jurisdiction_or_model} "{legal_entity_name}"')


@app.command()
def elf(
    jurisdiction_or_model: str,
    legal_name: str,
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, exists=True, dir_okay=True, resolve_path=True
    ),
    models_dir: Path = typer.Option(
        DEFAULT_MODEL_DIR, exists=True, dir_okay=True, resolve_path=True
    ),
):
    """
    Detect ELF codes for a Jurisdiction and legal name. Example: `lenu elf DE "Siemens AG"`
    """
    data_repo = DataRepo.from_data_dir(data_dir)

    if not data_repo.ready():
        logger.error("LEI data is not ready yet, Please use `lenu download`")
        sys.exit(1)

    model_repo = ModelRepo.from_models_dir(models_dir)

    huggingface_models = get_available_lenu_models_from_huggingface()
    fallback_model = f"Sociovestix/lenu_{jurisdiction_or_model}"

    try:
        elf_model = model_repo.get_model(jurisdiction_or_model)
        echo("Using locally trained ELF Detection model: " + jurisdiction_or_model)
        if fallback_model in huggingface_models:
            echo("We recommend using Transformer based model though: " + fallback_model)
    except ValueError as ve:
        if jurisdiction_or_model in huggingface_models:
            echo(
                f"Using recommended ELF Detection model from https://huggingface.co/{jurisdiction_or_model}"
            )
            elf_model = get_model_from_huggingface(jurisdiction_or_model)
        elif fallback_model in huggingface_models:
            echo(
                f"ELF Detection model for given jurisdiction {jurisdiction_or_model} not locally available."
            )
            echo(f"Using recommended model: https://huggingface.co/{fallback_model}")
            elf_model = get_model_from_huggingface(fallback_model)
        else:
            echo(
                f"ELF Detection model for provided jurisdiction '{jurisdiction_or_model}' does neither exist locally, nor is it available on https://huggingface.co/Sociovestix"
            )
            echo("")
            echo("You may train a scikit-learn based model locally. Example:")
            echo(f"lenu train DE")
            sys.exit(1)

    elf_probabilities = elf_model.detect(legal_name, top=3)

    # map things back to ELF Code and present
    elf_codes_names = data_repo.load_elf_code_list().get_names()
    res = (
        elf_codes_names.loc[elf_probabilities.index]
        .assign(Score=elf_probabilities)
        .reset_index()
        .rename(columns={"index": "ELF Code"})[
            ["ELF Code", "Entity Legal Form name Local name", "Score"]
        ]
    )

    jurisdiction = jurisdiction_or_model.split("_")[-1]

    echo("")
    echo(f'=== Top 3 ELF Codes in {jurisdiction} for "{legal_name}" ===')
    echo(res)


@app.command()
def abbreviations(
    jurisdiction: str,
    abbr: Optional[str] = None,
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, exists=True, dir_okay=True, resolve_path=True
    ),
):
    """
    List ELF Code legal form abbreviations for a Jurisdiction.
    """
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
