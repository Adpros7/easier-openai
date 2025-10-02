import base64
from binascii import Error
import os
from typing import Any, Literal
from urllib.parse import urlparse
from easier_openai import Assistant

class Openai_Images(Assistant):

    def __init__(self, image: str):
        """Parameters:
        image (str): The image to use for OpenAI API requests Can be Base64, Filepath, or URL.
        """

        
        def _classify_input(
            s: str,
        ) -> Literal["image_url", "Base64", "filepath", "unknown"]:
            # Check URL
            parsed = urlparse(s)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                return "image_url"

            # Check Base64
            try:
                decoded = base64.b64decode(s, validate=True)
                reencoded = base64.b64encode(decoded).decode("utf-8").rstrip("=")
                stripped = s.rstrip("=")
                if reencoded == stripped:
                    return "Base64"
            except (Error, ValueError):
                pass

            # Check Filepath
            if os.path.isfile(s):
                return "filepath"

            return "unknown"

        self.image = [image, "type", Any]
        self.type = _classify_input(image)
        if self.type == "unknown":
            raise ValueError("Image must be a Base64, Filepath, or URL.")

        else:
            self.image[1] = self.type
            if self.type == "Base64":
                self.image[2] = image.split(".")[1].lower().removeprefix(".")

                if self.image[2] not in ("jpg", "jpeg", "png", "gif", "webp"):
                    raise ValueError("Image must be a JPEG, PNG, GIF, or WEBP.")

                else:
                    if self.image[2] == "jpg":
                        self.image[2] = "jpeg"

            if self.type == "filepath":
                with open(self.image[0], "rb") as f:
                    file = self.client.files.create(file=f, purpose="vision")

                self.image[2] = file
