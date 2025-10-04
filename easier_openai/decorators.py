import re
import types
import inspect


def add_args_property(func: types.FunctionType) -> types.FunctionType:
    """
    Adds `.args` dict to a function based on 'Args:' or 'Params:' blocks in its docstring.
    Each entry: {name: {"description": str, "required": bool}}
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
        result = {}
        for line in lines:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            result[key.strip()] = val.strip()
        return result

    # Parse Args and Params
    args = extract_block("Args")
    params = extract_block("Params")
    merged = {**args, **params}

    # Use signature to figure out required vs optional
    sig = inspect.signature(func)
    structured = {}
    for name, desc in merged.items():
        param = sig.parameters.get(name)
        required = param.default is inspect._empty if param else True
        structured[name] = {"description": desc, "required": required}

    func.args = structured
    return func


# Example
@add_args_property
def test_func(a, b, c=0):
    """
    Does something.

    Args:
        a: The first arg.
        b: The second arg.

    Params:
        c: Optional thing.
    """
    return a + b + c


if __name__ == "__main__":
    print(test_func.args)
    # {
    #   'a': {'description': 'The first arg.', 'required': True},
    #   'b': {'description': 'The second arg.', 'required': True},
    #   'c': {'description': 'Optional thing.', 'required': False}
    # }

