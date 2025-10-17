from typing import Literal, TypeAlias
from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyCbeRUC1O95g5zfp1vdXE3JfvMZnLatNiU",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

models = client.models.list()
modler = []
# make it so that it only prints the model names
for model in models:
    modler.append(model.id.removeprefix("models/"))
    


gemini_models = Literal[*modler]

print(gemini_models)
    

