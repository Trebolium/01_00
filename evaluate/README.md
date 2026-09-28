# Evaluation

Manual evaluation scripts go here. For a 1-hour build the unit tests in `test/` are the primary
safety net; this directory is a placeholder for later scripts like:

- Feeding a set of known WAV files through `olympia.asr.transcriber` and checking WER
- Sanity-checking `olympia.biometrics.analyzer` output against a labeled reference clip
- End-to-end latency measurement across the full pipeline
