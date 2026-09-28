# Sends the transcript + biometric context to an LLM via OpenRouter and returns its reply.
import logging

from openai import OpenAI

from olympia.config import LLM_MODEL, LLM_SYSTEM_PROMPT, OPENROUTER_API_KEY
from olympia.types import BiometricFeatures

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _format_user_message(transcript_text: str, features: BiometricFeatures) -> str:
    # Prefix the transcript with a clearly labeled voice-metrics context block.
    metrics = (
        f"[Voice metrics: words-per-minute={features.words_per_minute}, "
        f"average pitch={features.avg_pitch_hz}hz, "
        f"average loudness={features.avg_loudness_db}db]"
    )
    return f"{metrics}\n\n{transcript_text}"


def get_response(transcript_text: str, features: BiometricFeatures) -> str:
    # Call OpenRouter (OpenAI-compatible API) with system + user messages, return the reply text.
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add OPENROUTER_API_KEY=<your-key> to your .env file "
            "(get a key at https://openrouter.ai/keys)."
        )

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    user_message = _format_user_message(transcript_text, features)

    logger.info("LLM request | model=%s | system=%r | user=%r", LLM_MODEL, LLM_SYSTEM_PROMPT, user_message)
    print(f"[llm] sending prompt to {LLM_MODEL}...")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    reply = response.choices[0].message.content

    logger.info("LLM response | %r", reply)
    print(f"[llm] response: {reply}")

    return reply
