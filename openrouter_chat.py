import streamlit as st
from openai import OpenAI


# -------------------------------
# Get API Key from Streamlit Secrets
# -------------------------------
def get_openrouter_api_key():
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return None


def validate_api_key():
    """Verify that the key exists and that OpenRouter accepts it.
    Returns a tuple ``(valid: bool, message: Optional[str])``. ``message``
    will contain an error or guidance if ``valid`` is False.
    """

    api_key = get_openrouter_api_key()
    if not api_key:
        return False, "OPENROUTER_API_KEY not set in Streamlit secrets."

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        # simple harmless request - list models without extra parameters
        client.models.list()
        return True, None
    except Exception as e:
        err = str(e)
        if "401" in err or "User not found" in err:
            return (
                False,
                (
                    "Authentication failed (401). Your OpenRouter API key may be "
                    "invalid, inactive, or not enabled for chat models. "
                    "Please verify your key at https://openrouter.ai/keys"
                ),
            )
        return False, f"Error validating API key: {err}"


# -------------------------------
# Chat Function
# -------------------------------


def chat_with_openrouter(prompt, history=None):

    api_key = get_openrouter_api_key()

    if not api_key:
        return "Error: OPENROUTER_API_KEY not set in Streamlit secrets."

    try:
        # Create OpenAI client for OpenRouter
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # Prepare messages
        messages = history[:] if history else []
        messages.append({"role": "user", "content": prompt})

        # Call OpenRouter API
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",  # Safe & recommended model
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e)

        if "401" in error_msg or "User not found" in error_msg:
            return (
                "Authentication failed (401). "
                "Your OpenRouter API key may be invalid, inactive, "
                "or not enabled for chat models. "
                "Please verify your key at https://openrouter.ai/keys"
            )

        return f"Error: {error_msg}"
