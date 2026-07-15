from os import chdir
from pathlib import Path
from openai import OpenAI

chdir(Path(__file__).resolve().parent)
models = [i.id for i in OpenAI().models.list()]

CODE = f"""from typing import TypeAlias, Literal

Model: TypeAlias = Literal{models.__repr__()}"""

Path("models.py").write_text(CODE)
