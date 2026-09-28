# Central config — constants and env vars used across modules.
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

SAMPLE_RATE = 16000
WAKE_PHRASE = "hey olympia"

# VAD: stop recording after this much consecutive silence mid-utterance
VAD_TRAILING_SILENCE_MS = 700

# Conversation loop: end the whole conversation if no voice detected within this long
CONVERSATION_SILENCE_TIMEOUT_S = 5.0

ASR_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "google/gemini-3.8-flash"

LLM_SYSTEM_PROMPT = (
    "You are a Voice Assistant designed to listen to and help the elderly. "
    "The provided prompt contains something your user said, accompanied by voice "
    "analysis feature metrics such as average pitch, average loudness and "
    "words-per-minute. Use these features to assist you in predicting the "
    "emotional state of your user, and respond to them appropriately. Respond "
    "with only a sentence or two, unless the question/instruction requires more "
    "detail."
)
