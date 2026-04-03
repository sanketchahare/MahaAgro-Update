import httpx
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Default offline responses for when API is unavailable
OFFLINE_RESPONSES = {
    "crop disease": [
        "Based on agricultural extension guidelines, common crop diseases include early blight, late blight, and bacterial spot. Professional diagnosis requires uploading a leaf image for AI analysis.",
        "Disease management typically involves: 1) Remove affected plants, 2) Apply appropriate fungicides, 3) Improve field drainage, 4) Rotate crops. Use the Crop Health tab for detailed analysis.",
    ],
    "pest control": [
        "Common pest management strategies: 1) Use pheromone traps for monitoring, 2) Apply neem oil or bio-pesticides, 3) Maintain field hygiene, 4) Rotate crops. Check the Pest Risk section for crop-specific advice.",
        "Recommended organic methods: Neem oil (3-5ml/L), Spinosad, or Bt spray. For chemical control, consult your local agricultural extension officer.",
    ],
    "irrigation": [
        "Irrigation depends on crop type, soil type, and weather. Generally: Cotton/corn need 500-1000mm/season, Rice needs 1000-1800mm. Use the Irrigation tab for personalized recommendations.",
        "Best practice: Water early morning (5-8 AM) when evaporation is lowest. Monitor soil moisture before irrigating. Drip irrigation saves 40-60% water.",
    ],
    "fertilizer": [
        "Soil testing is essential before fertilizer application. Get a soil test done at your nearest Krishi Vigyan Kendra (KVK). Use the Soil Health tab to upload your results.",
        "Basic fertilizer formula: Urea for Nitrogen, DAP for Phosphorus, MOP for Potassium. Application rates depend on soil testing results.",
    ],
    "weather": [
        "Weather significantly impacts crop management. Monitor temperature, humidity, and rainfall. High humidity increases disease risk. The Weather tab provides real-time forecasts.",
        "Extreme weather tips: Prepare for floods (improve drainage), droughts (irrigation planning), and storms (use windbreaks).",
    ],
    "default": [
        "I can help with crop diseases, pest control, irrigation, fertilizer, and soil management. Please upload a crop image or select your analysis parameters in the tabs above.",
        "For immediate agricultural assistance, call the Kisan Helpline: 1800-180-1551 (Toll-free in Maharashtra).",
    ],
}


# -------------------------------
# Get API Key from Streamlit Secrets
# -------------------------------
def get_openrouter_api_key():
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
        if key and key.strip() and not key.startswith("your_"):
            return key
        return None
    except Exception:
        return None


def is_api_available():
    """Check if OpenRouter API key is configured and valid."""
    api_key = get_openrouter_api_key()
    return api_key is not None


def validate_api_key():
    """Verify that the key exists and that OpenRouter accepts it."""
    api_key = get_openrouter_api_key()
    if not api_key:
        return (
            False,
            "OPENROUTER_API_KEY not configured. Using offline mode with built-in responses.",
        )

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MahaAgroAI/2.0",
        }
        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=5,
        )
        if response.status_code == 401:
            logger.warning("OpenRouter API authentication failed (401)")
            return False, (
                "OpenRouter API key authentication failed. "
                "Falling back to offline mode using built-in agricultural knowledge."
            )
        elif response.status_code == 200:
            return True, "OpenRouter API connected successfully"
        else:
            logger.warning(f"OpenRouter API returned status {response.status_code}")
            return False, (
                f"OpenRouter API error (status {response.status_code}). "
                "Using offline mode."
            )
    except httpx.TimeoutException:
        logger.warning("OpenRouter API timeout")
        return (
            False,
            "OpenRouter API timeout. Using offline mode with built-in responses.",
        )
    except Exception as e:
        logger.warning(f"Error validating API key: {str(e)}")
        return False, f"Using offline mode: {str(e)}"


# -------------------------------
# Offline Chat Function (Fallback)
# -------------------------------
def chat_offline(prompt):
    """Generate response using built-in agricultural knowledge."""
    prompt_lower = prompt.lower()
    
    for keyword, responses in OFFLINE_RESPONSES.items():
        if keyword in prompt_lower or keyword in prompt_lower.replace(" ", ""):
            import random
            return random.choice(responses)
    
    # Default response
    import random
    return random.choice(OFFLINE_RESPONSES["default"])


# -------------------------------
# Chat Function with Fallback
# -------------------------------
def chat_with_openrouter(prompt, history=None):
    """Chat with OpenRouter API, fallback to offline mode if API unavailable."""
    api_key = get_openrouter_api_key()

    # Fallback to offline mode if no API key
    if not api_key:
        logger.info("OpenRouter API key not configured, using offline mode")
        return chat_offline(prompt)

    try:
        messages = history[:] if history else []
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MahaAgroAI/2.0",
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
        }

        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )

        if response.status_code == 401:
            logger.warning("OpenRouter authentication failed, falling back to offline")
            return chat_offline(prompt)
        
        if response.status_code != 200:
            logger.warning(
                f"OpenRouter API error {response.status_code}, falling back to offline"
            )
            return chat_offline(prompt)

        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return chat_offline(prompt)

    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning(f"API timeout/connection error, using offline mode: {str(e)}")
        return chat_offline(prompt)
    except Exception as e:
        logger.warning(f"API error, falling back to offline mode: {str(e)}")
        return chat_offline(prompt)
