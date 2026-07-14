from os import mkdir
import httpx
from os.path import exists
from pathlib import Path


mkdir("")

files = ["Terminal.DIPAEiD0.js", "Search.DqJu-Pk0.js", "Desktop.CRMYUmUT.js", "model-recommendations.react.BiUt_81s.js", "navigation.react.B17za4aM.js", "jsx-runtime.u17CrQMm.js", "index.CzFgSF8h.js"]

for JS_FILE in files:
    if not exists(JS_FILE):
        with open(JS_FILE, "w", encoding="utf-8") as f:
            f.write(
                httpx.get(
                    f"https://developers.openai.com/_astro/{JS_FILE}"
                ).text
            )


