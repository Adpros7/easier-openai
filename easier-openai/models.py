import httpx
from os.path import exists
import os
import json
import re
from pathlib import Path

# Path to the downloaded JS bundle
JS_FILE = "models-page-data.react.BcABuaNa.js"

if not exists(JS_FILE):
    with open(JS_FILE, "w", encoding="utf-8") as f:
        f.write(httpx.get("https://developers.openai.com/_astro/models-page-data.react.BcABuaNa.js").text)

text = Path(JS_FILE).read_text(encoding="utf-8")

# Matches:
# var X={name:"gpt-5.5", ... supported_endpoints:["responses","batch"] ...}
pattern = re.compile(
    r'name:"([^"]+)".*?supported_endpoints:\[([^\]]*)\]',
    re.DOTALL,
)

result = {}

for model, endpoints in pattern.findall(text):
    eps = re.findall(r'"([^"]+)"', endpoints)
    result[model] = eps

print(json.dumps(result, indent=2, sort_keys=True))