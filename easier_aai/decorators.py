import re
import types
import inspect
from assistant import Assistant


def openai_function(func: types.FunctionType) -> dict:
    """
    Converts a plain function into a structured JSON-like schema
    derived from its docstring (Args:, Params:, Description:).

    Returns a dict like:
    {
        "type": "function",
        "name": "function_name",
        "description": "Short description",
        "parameters": {
            "type": "object",
            "properties": {
                "arg": {"type": "string", "description": "..."},
            },
            "required": ["arg"]
        }
    }

    Example:
        >>> @openai_function
        ... def greet(name: str) -> dict:
        ...     \"\"\"Description:\\n    Send a friendly greeting.\\n    Args:\\n        name: Who to greet.\\n    \"\"\"
        ...     return {\"message\": f\"Hello {name}!\"}
        >>> greet.schema["parameters"]["required"]
        ['name']

    Note:
        The decorator mutates ``func`` in-place by attaching a ``schema`` attribute.
    """
    if not isinstance(func, types.FunctionType):
        raise TypeError("Expected a plain function (types.FunctionType)")

    doc = inspect.getdoc(func) or ""

    def extract_block(name: str) -> dict:
        """Parse a docstring section (such as Args) into a dictionary for schema building.

        Args:
            name: Heading label to extract from the docstring.

        Returns:
            dict: Mapping of argument names to their descriptions.

        Example:
            If the wrapped docstring contains::

                Args:
                    city: Name of the city.

            then calling ``extract_block("Args")`` produces ``{"city": "Name of the city."}``.

        Note:
            Missing sections produce an empty dict so downstream merges remain straightforward.
        """
        pattern = re.compile(
            rf"{name}:\s*\n((?:\s+.+\n?)+?)(?=^[A-Z][A-Za-z_ ]*:\s*$|$)", re.MULTILINE
        )
        match = pattern.search(doc)
        if not match:
            return {}
        lines = match.group(1).strip().splitlines()
        block_dict = {}
        for line in lines:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            block_dict[key.strip()] = val.strip()
        return block_dict

    def extract_description() -> str:
        """Retrieve the free-form description block from the docstring if present.

        Example:
            Given a section such as::

                Description:
                    Provide a short overview.

            the helper returns ``"Provide a short overview."``.

        Note:
            Falls back to an empty string when no formatted section is found.
        """
        pattern = re.compile(
            r"Description:\s*\n((?:\s+.+\n?)+?)(?=^[A-Z][A-Za-z_ ]*:\s*$|$)",
            re.MULTILINE,
        )
        match = pattern.search(doc)
        if not match:
            return ""
        return " ".join(line.strip() for line in match.group(1).splitlines())

    args = extract_block("Args")
    params = extract_block("Params")
    merged = {**args, **params}
    description = extract_description()

    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, desc in merged.items():
        param = sig.parameters.get(name)
        required_flag = param.default is inspect._empty if param else True
        properties[name] = {
            "type": "string",  # you could infer more types if needed
            "description": desc,
        }
        if required_flag:
            required.append(name)

    schema = {
        "type": "function",
        "name": func.__name__,
        "description": description or func.__doc__.strip().split("\n")[0], # type: ignore
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    func.schema = schema
    return func # type: ignore
