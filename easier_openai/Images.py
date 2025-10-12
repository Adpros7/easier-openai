import base64
from binascii import Error
import os
from typing import Any, Literal
from urllib.parse import urlparse
from easier_openai import Assistant


class Openai_Images(Assistant):
    """Simplified interface for classifying and uploading image payloads to OpenAI.

    Example:
        >>> helper = Openai_Images("https://example.com/cat.png")  # doctest: +SKIP
        >>> helper.type
        'image_url'

    Note:
        Inherits all assistant capabilities so you can immediately call vision-enabled methods.
    """

    def __init__(self, image: str):
        """Initialise the helper and classify the provided image reference.

        Args:
            image: Base64 content, filesystem path, or HTTP(S) URL pointing to an image.

        Example:
            >>> helper = Openai_Images("samples/photo.jpg")  # doctest: +SKIP
            >>> helper.image[1]
            'filepath'

        Note:
            The parent ``Assistant`` initialises with default arguments; adjust attributes after construction if necessary.
        """

        super().__init__()

        def _classify_input(
            s: str,
        ) -> Literal["image_url", "Base64", "filepath", "unknown"]:
            """Infer the type of image input so downstream calls can handle it appropriately.

            Args:
                s: Raw image value supplied by the caller.

            Returns:
                Literal[str]: One of ``\"image_url\"``, ``\"Base64\"``, ``\"filepath\"``,
                or ``\"unknown\"`` indicating how the image should be processed.

            Example:
                >>> _classify_input("https://example.com/cat.png")  # doctest: +SKIP
                'image_url'
                >>> _classify_input("/tmp/cat.png")  # doctest: +SKIP
                'filepath'

            Note:
                Inputs that fail all checks surface as ``\"unknown\"`` so callers can raise helpful errors.
            """
            # Check URL
            parsed = urlparse(s)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                return "image_url"

            # Check Base64
            try:
                decoded = base64.b64decode(s=s, validate=True)
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
                    file = self._client.files.create(file=f, purpose="vision")

                self.image[2] = file.id
