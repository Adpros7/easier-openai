import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from easier_openai.assistant import Assistant


def make_assistant() -> Assistant:
    """Return an Assistant instance with network calls stubbed for isolated testing.

    Example:
        >>> assistant = make_assistant()
        >>> assistant._model
        'test-model'
    """
    assistant = Assistant.__new__(Assistant)
    assistant._system_prompt = ""
    assistant._model = "test-model"
    assistant._reasoning = None
    assistant._temperature = None
    assistant._conversation_id = "conv_test"
    assistant._conversation = SimpleNamespace(id="conv_test")
    assistant._function_call_list = []
    assistant._client = SimpleNamespace(responses=MagicMock())
    return assistant


def test_prepare_function_tools_generates_schema_and_map():
    """Ensure tool preparation auto-decorates callables and builds schema + lookup table.

    Example:
        >>> assistant = make_assistant()
        >>> prepared, tool_map = assistant._prepare_function_tools([])
        >>> isinstance(prepared, list) and isinstance(tool_map, dict)
        True
    """
    assistant = make_assistant()

    def describe_city(city: str) -> dict:
        """Provide a structured description payload for a requested city.

        Description:
            Provide a short description for a city.
        Args:
            city: The city to describe.

        Example:
            >>> describe_city("Paris")
            {'city': 'Paris'}
        """

        return {"city": city}

    prepared, tool_map = assistant._prepare_function_tools([describe_city])

    assert len(prepared) == 1
    schema = prepared[0]
    assert schema["type"] == "function"
    assert schema["name"] == "describe_city"
    assert "city" in schema["parameters"]["properties"]
    assert tool_map["describe_city"] is describe_city
    assert hasattr(describe_city, "schema")


def test_resolve_response_with_tools_executes_tools_and_returns_final_response():
    """The helper should execute tool calls until all function calls are fulfilled.

    Example:
        >>> assistant = make_assistant()
        >>> callable(assistant._resolve_response_with_tools)
        True
    """
    assistant = make_assistant()

    func_call_item = SimpleNamespace(
        type="function_call",
        name="describe_city",
        arguments=json.dumps({"city": "Paris"}),
        call_id="call_1",
    )

    initial_response = SimpleNamespace(
        output=[func_call_item],
        output_text="",
        conversation=SimpleNamespace(id="conv_test"),
    )

    final_response = SimpleNamespace(
        output=[],
        output_text="done",
        conversation=SimpleNamespace(id="conv_test"),
    )

    assistant._client.responses.create.side_effect = [initial_response, final_response]

    def describe_city(city: str) -> dict:
        """Return a simple city payload for testing tool execution.

        Example:
            >>> describe_city("Paris")
            {'city': 'Paris'}
        """
        return {"city": city}

    describe_city = assistant.openai_function(describe_city)

    params = {
        "model": "test-model",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
        ],
        "conversation": "conv_test",
        "tools": [describe_city.schema],
    }

    result = assistant._resolve_response_with_tools(
        params, {"describe_city": describe_city}
    )

    assert result is final_response
    assert assistant._client.responses.create.call_count == 2
    second_call_kwargs = assistant._client.responses.create.call_args_list[1].kwargs
    assert second_call_kwargs["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": json.dumps({"city": "Paris"}),
        }
    ]


def test_chat_flows_through_function_calling_cycle():
    """Verify the chat flow triggers tool execution and finalizes with the response.

    Example:
        >>> assistant = make_assistant()
        >>> hasattr(assistant, "chat")
        True
    """
    assistant = make_assistant()

    captured = {}

    def describe_city(city: str) -> dict:
        """Capture and normalize the city argument for asserting tool usage.

        Example:
            >>> describe_city("paris")
            {'city': 'Paris'}
        """
        captured["city"] = city
        return {"city": city.title()}

    func_call_item = SimpleNamespace(
        type="function_call",
        name="describe_city",
        arguments=json.dumps({"city": "paris"}),
        call_id="call_1",
    )

    initial_response = SimpleNamespace(
        output=[func_call_item],
        output_text="",
        conversation=SimpleNamespace(id="conv_test"),
    )

    final_response = SimpleNamespace(
        output=[],
        output_text="Paris is lovely.",
        conversation=SimpleNamespace(id="conv_test"),
    )

    assistant._client.responses.create.side_effect = [initial_response, final_response]

    result = assistant.chat("Tell me about Paris", custom_tools=[describe_city])

    assert result == "Paris is lovely."
    assert assistant._client.responses.create.call_count == 2
    first_call_kwargs = assistant._client.responses.create.call_args_list[0].kwargs
    tools_arg = first_call_kwargs["tools"]
    assert tools_arg and tools_arg[0]["name"] == "describe_city"
    second_call_kwargs = assistant._client.responses.create.call_args_list[1].kwargs
    assert second_call_kwargs["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": json.dumps({"city": "Paris"}),
        }
    ]
    assert captured["city"] == "paris"
