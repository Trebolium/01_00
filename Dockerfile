FROM python:3.11-slim

# portaudio for mic I/O (sounddevice), libsndfile for audio decoding (librosa/soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# openwakeword ships no model weights in its pip package — bake them into the image at build
# time so the container doesn't need network access on every start to fetch them.
RUN uv run python -c "import openwakeword; openwakeword.utils.download_models()"

COPY src ./src

CMD ["uv", "run", "python", "-m", "olympia.orchestrator"]
