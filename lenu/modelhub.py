import pandas
import requests
from transformers import pipeline


def get_available_lenu_models_from_huggingface():
    r = requests.get(
        "https://huggingface.co/api/models", params={"search": "Sociovestix"}
    )

    return list(
        sorted(
            [
                model["modelId"]
                for model in r.json()
                if (
                    not model["private"]
                    and model["modelId"].startswith("Sociovestix/lenu_")
                )
            ]
        )
    )


class ELFDetectionModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def detect(self, legal_name, top=3):
        # do the prediction
        elf_probabilities = pandas.Series(
            {
                res["label"][0:4]: res["score"]
                for res in self.pipeline(legal_name, top_k=top)
            }
        )
        return elf_probabilities


def get_model_from_huggingface(repo_name):
    pipe = pipeline(model=repo_name)
    return ELFDetectionModel(pipe)
