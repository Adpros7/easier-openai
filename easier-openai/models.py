import subprocess
import json
from os import mkdir, rename
import httpx
from os.path import exists
from pathlib import Path



files = ["Terminal.DIPAEiD0.js", "Search.DqJu-Pk0.js", "Desktop.CRMYUmUT.js", "model-recommendations.react.BiUt_81s.js", "navigation.react.B17za4aM.js", "jsx-runtime.u17CrQMm.js", "index.CzFgSF8h.js", "models-page-data.react.BcABuaNa.js"]

for JS_FILE in files:
    if not exists(JS_FILE):
        with open(JS_FILE, "w", encoding="utf-8") as f:
            f.write(
                httpx.get(
                    f"https://developers.openai.com/_astro/{JS_FILE}"
                ).text
            )


rename("models-page-data.react.BcABuaNa.js", "models.js")

NODE_DIR = (Path(__file__).parent / "_node").resolve()

out = subprocess.run(
    ["node", str(NODE_DIR / "endpoints.js"), ], capture_output=True
)

