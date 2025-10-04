import re
import types
import inspect
from assistant import Assistant

def add_args_property(func: types.FunctionType) -> types.FunctionType:
    """
    Adds `.args` dict to a function based on its docstring's 'Args:' or 'Params:' sections.
    Only works on plain functions (types.FunctionType).
    """
    if not isinstance(func, types.FunctionType):
        raise TypeError("Expected a plain function (types.FunctionType)")

    doc = inspect.getdoc(func) or ""

    def extract_block(name: str) -> dict:
        # Capture block up to the next section header or end of string
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

    args = extract_block("Args")
    params = extract_block("Params")

    func.args = {**args, **params}
    return func


# Example
add_args_property(Assistant.chat)


if __name__ == "__main__":
    print(Assistant.chat.args)
    # {'a': 'The first arg.', 'b': 'The second arg.', 'c': 'Optional thing.'}
