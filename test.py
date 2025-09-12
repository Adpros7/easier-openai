import base64

data = "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zUAAAA..."  # your full string
with open("output.png", "wb") as f:
    f.write(base64.b64decode(data))
