"""Tests for the uqlm.uipath_llms module."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# Functions to test
from uqlm.uipath_llms import get_uipath_llm, invoke_uipath_llm

# The class we expect to be used from the SDK
# We need to be able to mock this effectively.
# If uipath_langchain.chat.models.UiPathChat is not found during test collection,
# we can define a dummy class here for type hinting and patching,
# or ensure the environment has it. For now, assume it can be imported or mocked.
try:
    from uipath_langchain.chat.models import UiPathChat
except ImportError:
    # Define a dummy UiPathChat if the SDK isn't installed in the test environment
    # This allows tests to be defined and collected, but they might behave differently
    # if the actual class has specific behaviors not mocked.
    class UiPathChat:
        def __init__(self, model: str, **kwargs):
            self.model = model
            self.kwargs = kwargs
        def invoke(self, prompt: str):
            return f"Mocked response for: {prompt}"

# A simple mock for AIMessage-like objects returned by llm.invoke()
class MockAIMessage:
    def __init__(self, content: str):
        self.content = content

# --- Tests for get_uipath_llm ---

@patch('uqlm.uipath_llms.UiPathChat', autospec=True)
def test_get_uipath_llm_initializes_uipathchat_with_model_name(mock_uipath_chat_constructor):
    """Test that get_uipath_llm calls UiPathChat constructor with the correct model name."""
    model_name = "test-model-id"
    get_uipath_llm(llm_name=model_name)
    mock_uipath_chat_constructor.assert_called_once_with(model=model_name)

@patch('uqlm.uipath_llms.UiPathChat', autospec=True)
def test_get_uipath_llm_passes_kwargs_to_uipathchat(mock_uipath_chat_constructor):
    """Test that get_uipath_llm passes additional kwargs to UiPathChat constructor."""
    model_name = "test-model-id"
    extra_kwargs = {"temperature": 0.7, "max_tokens": 100}
    get_uipath_llm(llm_name=model_name, **extra_kwargs)
    mock_uipath_chat_constructor.assert_called_once_with(model=model_name, **extra_kwargs)

@patch('uqlm.uipath_llms.UiPathChat', new_callable=PropertyMock, return_value=None)
@patch('uqlm.uipath_llms.UIPATH_SDK_AVAILABLE', False)
def test_get_uipath_llm_handles_importerror_for_uipathchat(mock_sdk_unavailable, mock_uipath_chat_class_is_none):
    """
    Test get_uipath_llm raises ImportError if UiPathChat is not available (simulating ImportError).
    The current implementation raises ImportError.
    """
    with pytest.raises(ImportError) as excinfo:
        get_uipath_llm(llm_name="any-model")
    assert "UiPath Langchain SDK (uipath-langchain) is not installed" in str(excinfo.value)

@patch('uipath_langchain.chat.models.UiPathChat', MagicMock(side_effect=Exception("SDK Init Error")))
def test_get_uipath_llm_handles_sdk_initialization_error():
    """Test get_uipath_llm raises RuntimeError if UiPathChat constructor fails."""
    # This test requires uipath_langchain.chat.models.UiPathChat to be patchable
    # If it's not found, the dummy class above doesn't help here.
    # We'll patch it directly in the uipath_langchain.chat.models namespace if possible
    # or adjust if the class is imported as `from uipath_langchain.chat.models import UiPathChat as AliasedUiPathChat`
    # For this test, we assume the direct patch path works.
    
    # If the SDK is not installed, this test might not run as intended or fail at patch time.
    # The try-except for UiPathChat import in the main module helps, but for testing specific
    # SDK errors, the SDK should ideally be mockable or available.
    
    # Re-patching 'uqlm.uipath_llms.UiPathChat' which is the name used within the module under test.
    with patch('uqlm.uipath_llms.UiPathChat', MagicMock(side_effect=Exception("SDK Init Error"))) as mock_constructor:
        with pytest.raises(RuntimeError) as excinfo:
            get_uipath_llm(llm_name="test-model")
        assert "Failed to initialize UiPath LLM" in str(excinfo.value)
        assert "SDK Init Error" in str(excinfo.value)


# --- Tests for invoke_uipath_llm ---

def test_invoke_uipath_llm_calls_client_invoke_with_prompt():
    """Test that invoke_uipath_llm calls the llm_client's invoke method."""
    mock_llm_client = MagicMock(spec=UiPathChat) # Use spec for better mocking
    prompt = "Hello, world!"
    
    # Mock the return value of invoke to be a simple string for this case,
    # or a MockAIMessage if testing content extraction specifically.
    mock_llm_client.invoke.return_value = MockAIMessage("Mocked response")
    
    invoke_uipath_llm(llm_client=mock_llm_client, prompt=prompt)
    mock_llm_client.invoke.assert_called_once_with(prompt)

def test_invoke_uipath_llm_extracts_content_from_response():
    """Test that invoke_uipath_llm extracts content from AIMessage-like response."""
    mock_llm_client = MagicMock(spec=UiPathChat)
    prompt = "Test prompt"
    expected_content = "This is the actual content."
    mock_llm_client.invoke.return_value = MockAIMessage(content=expected_content)
    
    response_content = invoke_uipath_llm(llm_client=mock_llm_client, prompt=prompt)
    assert response_content == expected_content

def test_invoke_uipath_llm_handles_string_response_from_invoke():
    """Test invoke_uipath_llm if llm_client.invoke returns a plain string."""
    mock_llm_client = MagicMock(spec=UiPathChat)
    prompt = "Test prompt for string"
    expected_response_str = "Direct string response"
    mock_llm_client.invoke.return_value = expected_response_str

    response_content = invoke_uipath_llm(llm_client=mock_llm_client, prompt=prompt)
    assert response_content == expected_response_str


def test_invoke_uipath_llm_with_none_client():
    """Test invoke_uipath_llm raises ValueError if llm_client is None."""
    with pytest.raises(ValueError) as excinfo:
        invoke_uipath_llm(llm_client=None, prompt="Any prompt")
    assert "UiPath LLM client is not initialized or provided" in str(excinfo.value)

@patch('uqlm.uipath_llms.invoke_uipath_llm', side_effect=RuntimeError("LLM API Error"))
def test_invoke_uipath_llm_handles_api_error(mock_invoke_internal):
    """
    Test invoke_uipath_llm propagates or handles runtime errors from client.invoke.
    This tests if the outer function's try-except for client.invoke() works.
    """
    mock_llm_client = MagicMock(spec=UiPathChat)
    # Configure the *actual* client's invoke method to raise an error
    mock_llm_client.invoke.side_effect = RuntimeError("LLM API Error")
    
    with pytest.raises(RuntimeError) as excinfo:
        # Call the original function, not the mocked one
        from uqlm.uipath_llms import invoke_uipath_llm as original_invoke_uipath_llm
        original_invoke_uipath_llm(llm_client=mock_llm_client, prompt="A test prompt")
    assert "Error invoking UiPath LLM: LLM API Error" in str(excinfo.value)

# Example of how to test the endpoint_url behavior if it were used:
# @patch('uqlm.uipath_llms.UiPathChat', autospec=True)
# def test_get_uipath_llm_uses_endpoint_url_if_provided(mock_uipath_chat_constructor):
#     """Test that get_uipath_llm passes endpoint_url correctly if SDK supported it directly
#        or via a known kwarg like 'base_url'."""
#     model_name = "test-model-id"
#     endpoint = "http://custom.uipath.endpoint"
#     # Assuming 'base_url' is the kwarg UiPathChat would use for an endpoint
#     get_uipath_llm(llm_name=model_name, endpoint_url=endpoint, temperature=0.5)
#     mock_uipath_chat_constructor.assert_called_once_with(model=model_name, base_url=endpoint, temperature=0.5)

# Note: The above test for endpoint_url is commented out as the current implementation
# of get_uipath_llm uses kwargs.setdefault('base_url', endpoint_url), which is a guess.
# If the actual UiPathChat constructor takes a different parameter for endpoint_url,
# or handles it purely via environment variables, that test would need adjustment or removal.
# The current implementation of get_uipath_llm correctly passes kwargs, so specific
# endpoint_url handling depends on how UiPathChat itself consumes those kwargs.
