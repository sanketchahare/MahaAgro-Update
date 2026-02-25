import streamlit as st
import openai
import os


# Load OpenRouter API key from Streamlit secrets
def get_openrouter_api_key():
    return st.secrets["OPENROUTER_API_KEY"]


def chat_with_openrouter(prompt, history=None):
    # ensure we have a key and return a friendly message otherwise
    api_key = get_openrouter_api_key()
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set in Streamlit secrets or environment."

    client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except openai.AuthenticationError as e:
        # catch the common 401 case and provide guidance
        # the OpenRouter service returns "User not found" when the key
        # itself is valid for listing models but not enabled for chat
        msg = str(e)
        if "User not found" in msg or e.status_code == 401:
            return (
                "Authentication failed: the provided OpenRouter API key is not enabled for chat. "
                "You may have a read‑only key or one tied to an account without chat access. "
                "Obtain a proper key at https://openrouter.ai/ or contact support."
            )
        return (
            "Authentication failed: your OpenRouter API key was rejected. "
            "Please double-check the value in Streamlit secrets or obtain a valid key at https://openrouter.ai/"
        )
    except Exception as e:
        # propagate other errors as text so UI can display them
        return f"Error: {e}"
