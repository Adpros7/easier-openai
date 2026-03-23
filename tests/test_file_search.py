import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from easier_openai.assistant import Assistant


def make_assistant() -> Assistant:
    """Return an Assistant instance with network calls stubbed for isolated testing."""
    assistant = Assistant.__new__(Assistant)
    assistant._system_prompt = ""
    assistant._model = "test-model"
    assistant._reasoning = None
    assistant._temperature = None
    assistant._conversation_id = "conv_test"
    assistant._conversation = SimpleNamespace(id="conv_test")
    assistant._function_call_list = []
    assistant._client = SimpleNamespace(
        responses=MagicMock(),
        vector_stores=MagicMock(),
    )
    return assistant


def test_convert_filepath_to_vector_creates_store_and_uploads_files():
    """_convert_filepath_to_vector should create a vector store and upload each file.

    Example:
        >>> assistant = make_assistant()
        >>> hasattr(assistant, '_convert_filepath_to_vector')
        True
    """
    assistant = make_assistant()

    fake_store = SimpleNamespace(id="vs_123", name="vector_store")
    assistant._client.vector_stores.create.return_value = fake_store

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"hello world")
        tmp_path = f.name

    try:
        store, manager = assistant._convert_filepath_to_vector([tmp_path])
    finally:
        os.unlink(tmp_path)

    assert store is fake_store
    assert manager is assistant._client.vector_stores
    assistant._client.vector_stores.create.assert_called_once_with(name="vector_store")
    assistant._client.vector_stores.files.upload_and_poll.assert_called_once()
    call_kwargs = assistant._client.vector_stores.files.upload_and_poll.call_args.kwargs
    assert call_kwargs["vector_store_id"] == "vs_123"


def test_convert_filepath_to_vector_raises_on_empty_list():
    """Should raise ValueError when given an empty file list.

    Example:
        >>> assistant = make_assistant()
        >>> try:
        ...     assistant._convert_filepath_to_vector([])
        ... except ValueError as e:
        ...     str(e)
        'list_of_files must be a non-empty list of file paths.'
    """
    assistant = make_assistant()
    try:
        assistant._convert_filepath_to_vector([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_convert_filepath_to_vector_raises_on_missing_file():
    """Should raise FileNotFoundError for paths that do not exist.

    Example:
        >>> assistant = make_assistant()
        >>> try:
        ...     assistant._convert_filepath_to_vector(['/nonexistent/path.txt'])
        ... except FileNotFoundError:
        ...     pass
    """
    assistant = make_assistant()
    try:
        assistant._convert_filepath_to_vector(["/nonexistent/path.txt"])
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_chat_with_file_search_passes_vector_store_ids_as_list():
    """chat() must pass vector_store_ids as a list, not a bare string.

    Example:
        >>> assistant = make_assistant()
        >>> hasattr(assistant, 'chat')
        True
    """
    assistant = make_assistant()

    fake_store = SimpleNamespace(id="vs_abc")
    assistant._client.vector_stores.create.return_value = fake_store

    final_response = SimpleNamespace(
        output=[],
        output_text="Here is what I found.",
        conversation=SimpleNamespace(id="conv_test"),
    )
    assistant._client.responses.create.return_value = final_response

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"relevant content")
        tmp_path = f.name

    try:
        result = assistant.chat("Search the file", file_search=[tmp_path])
    finally:
        os.unlink(tmp_path)

    assert result == "Here is what I found."

    call_kwargs = assistant._client.responses.create.call_args.kwargs
    tools = call_kwargs["tools"]
    file_search_tool = next(t for t in tools if t["type"] == "file_search")
    assert isinstance(file_search_tool["vector_store_ids"], list), (
        "vector_store_ids must be a list, got "
        + type(file_search_tool["vector_store_ids"]).__name__
    )
    assert file_search_tool["vector_store_ids"] == ["vs_abc"]


def test_chat_with_file_search_deletes_vector_store_after_response():
    """chat() should delete the temporary vector store once the response is returned.

    Example:
        >>> assistant = make_assistant()
        >>> hasattr(assistant, 'chat')
        True
    """
    assistant = make_assistant()

    fake_store = SimpleNamespace(id="vs_xyz")
    assistant._client.vector_stores.create.return_value = fake_store

    final_response = SimpleNamespace(
        output=[],
        output_text="done",
        conversation=SimpleNamespace(id="conv_test"),
    )
    assistant._client.responses.create.return_value = final_response

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"data")
        tmp_path = f.name

    try:
        assistant.chat("Find something", file_search=[tmp_path])
    finally:
        os.unlink(tmp_path)

    assistant._client.vector_stores.delete.assert_called_once_with("vs_xyz")


def test_chat_with_file_search_and_max_searches():
    """max_searches parameter should be forwarded to the file_search tool spec.

    Example:
        >>> assistant = make_assistant()
        >>> hasattr(assistant, 'chat')
        True
    """
    assistant = make_assistant()

    fake_store = SimpleNamespace(id="vs_max")
    assistant._client.vector_stores.create.return_value = fake_store

    final_response = SimpleNamespace(
        output=[],
        output_text="ok",
        conversation=SimpleNamespace(id="conv_test"),
    )
    assistant._client.responses.create.return_value = final_response

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"data")
        tmp_path = f.name

    try:
        assistant.chat(
            "Find something", file_search=[tmp_path], file_search_max_searches=5
        )
    finally:
        os.unlink(tmp_path)

    call_kwargs = assistant._client.responses.create.call_args.kwargs
    tools = call_kwargs["tools"]
    file_search_tool = next(t for t in tools if t["type"] == "file_search")
    assert file_search_tool["max_searches"] == 5
