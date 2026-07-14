import time
import os
import subprocess
import json
from os import rename
import httpx
from os.path import exists
from pathlib import Path
from platformdirs import user_cache_path


files = [
    "Terminal.DIPAEiD0.js",
    "Search.DqJu-Pk0.js",
    "Desktop.CRMYUmUT.js",
    "model-recommendations.react.BiUt_81s.js",
    "navigation.react.B17za4aM.js",
    "jsx-runtime.u17CrQMm.js",
    "index.CzFgSF8h.js",
    "models-page-data.react.BcABuaNa.js",
]

CACHE_DIR = user_cache_path() / "easier-openai"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
og_dir = Path().resolve()
os.chdir(CACHE_DIR)

INVALIDATE_CACHE_AFTER = 60 * 60 * 24

if not exists("time.log"):
    with open(CACHE_DIR / "time.log", "w") as f:
        f.write(str(time.time()))

for JS_FILE in files:
    if (
        not exists(JS_FILE)
        or float(Path("time.log").read_text()) - float(time.time())
        > INVALIDATE_CACHE_AFTER
    ):
        with open(JS_FILE, "w", encoding="utf-8") as f:
            f.write(httpx.get(f"https://developers.openai.com/_astro/{JS_FILE}").text)

rename("models-page-data.react.BcABuaNa.js", "models.js")

NODE_DIR = (Path(__file__).parent / "_node").resolve()

og_code = Path(NODE_DIR / "model_info.js").read_text()
Path("modelInfoRun.js").write_text(og_code)

def get_model_info(model: str):
    info = subprocess.run(
        ["node", "modelInfoRun.js", model],
        capture_output=True,
        text=True,
        cwd=CACHE_DIR,
    )

    return json.loads(info.stdout)