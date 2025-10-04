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
    """
    if not isinstance(func, types.FunctionType):
        raise TypeError("Expected a plain function (types.FunctionType)")

    doc = inspect.getdoc(func) or ""

    def extract_block(name: str) -> dict:
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
