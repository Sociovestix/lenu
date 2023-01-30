
import requests

def get_lenu_models_from_huggingface():
    r = requests.get("https://huggingface.co/api/models", params={
        "search": "Sociovestix"
    })
     
    return list(
        sorted(
            [model["modelId"] for model in r.json() if not model["private"]]
        )
    )
