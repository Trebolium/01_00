FROM python:3.11-slim

# portaudio for mic I/O (sounddevice), libsndfile for audio decoding (librosa/soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src

CMD ["uv", "run", "python", "-m", "olympia.orchestrator"]
