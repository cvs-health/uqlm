"""UiPath LLM integration for the UQLM package."""

from typing import Any

import os
from typing import Any, Optional

# Attempt to import UiPath LLM components.
# Based on documentation: https://uipath.github.io/uipath-python/langchain/chat_models/
try:
    from uipath_langchain.chat.models import UiPathChat
    # For specific Azure OpenAI models via UiPath, one could use UiPathAzureChatOpenAI
    # from uipath_langchain.chat.models import UiPathAzureChatOpenAI
    UIPATH_SDK_AVAILABLE = True
except ImportError:
    UiPathChat = None # type: ignore 
    UIPATH_SDK_AVAILABLE = False

def get_uipath_llm(
    llm_name: str, 
    endpoint_url: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """
    Initializes and returns a UiPath LLM client using the UiPathChat class.

    This function leverages the UiPath Langchain SDK to connect to UiPath-managed LLMs.
    Authentication is typically handled via environment variables configured through the
    UiPath CLI (`uipath auth`).

    Required Environment Variables (ensure these are set via `uipath auth` or manually):
    - UIPATH_URL: Your UiPath Automation Cloud URL (e.g., https://cloud.uipath.com/yourorg/yourtenant).
    - UIPATH_CLIENT_ID: Your UiPath application client ID (for OAuth).
    - UIPATH_CLIENT_SECRET: Your UiPath application client secret (for OAuth).
    OR
    - UIPATH_ACCESS_TOKEN: A direct access token (obtained via `uipath auth` or other means).
    
    Note: The `endpoint_url` parameter is included for flexibility but may not be directly
    used by `UiPathChat` if its configuration primarily relies on `UIPATH_URL` and other
    environment variables for service discovery. Refer to specific UiPath documentation
    if connecting to a non-standard AI Trust Layer or on-premise instance.

    Args:
        llm_name: The model identifier for the UiPath LLM to use
                  (e.g., "anthropic.claude-3-opus-20240229-v1:0", "gpt-4o-2024-05-13").
        endpoint_url: Optional. The specific endpoint URL for the UiPath AI Trust Layer
                      or LLM service. May be overridden by SDK's environment variable configuration.
        **kwargs: Additional keyword arguments to pass to the UiPathChat constructor
                  (e.g., temperature, max_tokens).

    Returns:
        An initialized UiPathChat client.

    Raises:
        ImportError: If the `uipath-langchain` SDK is not installed.
        RuntimeError: If initialization fails (e.g., due to missing credentials or
                      issues connecting to the UiPath platform).
    """
    if not UIPATH_SDK_AVAILABLE or UiPathChat is None:
        raise ImportError(
            "UiPath Langchain SDK (uipath-langchain) is not installed or UiPathChat could not be imported. "
            "Please install it using: pip install uipath-langchain"
        )

    try:
        # The UiPathChat constructor primarily uses `model` and other standard LLM params.
        # Authentication and endpoint are typically picked up from environment variables
        # set by `uipath auth` (UIPATH_URL, UIPATH_ACCESS_TOKEN or OAuth credentials).
        # If `endpoint_url` is provided and the SDK supports it directly, it could be passed
        # here, but current documentation for UiPathChat doesn't show it as a direct param.
        # We will pass through kwargs for flexibility.
        if endpoint_url:
            # This is a common pattern if an SDK supports overriding the base URL.
            # However, verify with UiPath specific documentation if this is the correct way.
            # For now, we assume it might be part of kwargs or handled by env vars.
            kwargs.setdefault('base_url', endpoint_url) # Example, may not be correct for UiPathChat
            # Or, more generically, let it be in kwargs if the user knows the specific param name.

        llm_client = UiPathChat(model=llm_name, **kwargs)
        return llm_client
    except Exception as e:
        # Catching a broad exception here as specific SDK errors are not documented yet.
        # Ideally, catch specific exceptions like UiPathAuthError, UiPathConnectionError if they exist.
        raise RuntimeError(
            f"Failed to initialize UiPath LLM (model: {llm_name}). "
            f"Ensure environment variables (UIPATH_URL, UIPATH_ACCESS_TOKEN or OAuth credentials) are correctly set. "
            f"Original error: {e}"
        )

def invoke_uipath_llm(llm_client: Any, prompt: str) -> str:
    """
    Invokes the initialized UiPath LLM with a given prompt and returns the response.

    Args:
        llm_client: An initialized UiPath LLM client (e.g., an instance of `UiPathChat`).
        prompt: The prompt string to send to the LLM.

    Returns:
        The LLM's response as a string.

    Raises:
        ValueError: If the `llm_client` is not provided or is None.
        RuntimeError: If an error occurs during LLM invocation.
    """
    if llm_client is None:
        raise ValueError("UiPath LLM client is not initialized or provided.")

    try:
        response = llm_client.invoke(prompt)
        # Langchain LLM .invoke() typically returns an AIMessage or similar object.
        # The actual text content is usually in a 'content' attribute.
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response) # Fallback if it's already a string or different object type
    except Exception as e:
        # Catching a broad exception. Replace with specific SDK errors if known.
        raise RuntimeError(f"Error invoking UiPath LLM: {e}")
