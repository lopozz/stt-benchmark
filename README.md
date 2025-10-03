# stt-benchmark
Benchmarking scripts to evaluate STT models as as server

# Setup
Prepare the venv:
```
python3 -m venv .venv
source .venv/bin/activate
pip install pip-tools  -r requirements.txt
```

Build the docker image:
```
docker compose build
```

Add your audio data to `data`.

```
docker run --rm -it --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-openai-audio:latest \
  --model openai/whisper-large-v3-turbo \
  --task transcription \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 128 \
  --max-num-seqs 2
```

Check the server is working with a simple call:
```
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@path/to/file.wav" \
  -F "model=openai/whisper-large-v3-turbo"
```

## CLI Benchmark (per-user sequential requests)
This script drives your /v1/audio/transcriptions endpoint and measures latency/throughput.
It models c concurrent users, each sending n sequential requests (no client-side pipelining).
So for a given concurrency `c` and requests-per-user `n`, total requests = c × n.

#### Usage:
```
python bench_seq.py \
  --url "http://localhost:8000/v1/audio/transcriptions" \
  -m openai/whisper-large-v3-turbo \
  -f ./samples/jfk.wav \
  -n 3 \
  -c 1 2 4 8 16 \
  --language it \
  --task transcribe \
  --output json

```

## Batch Transcription
This script scans a directory for audio files and transcribes each one by POSTing to an OpenAI-compatible
/v1/audio/transcriptions endpoint. If the server rejects a file for being too long (e.g., >30s), the script applies a [**Silero VAD**](https://github.com/snakers4/silero-vad) fallback to split speech into chunks ≤ 29.5s and stitches the partial transcripts.

#### Usage:
```
python batch_transcribe.py \
  --url "http://localhost:8000/v1/audio/transcriptions" \
  --indir ./data/audio \
  --outdir ./transcripts \
  --model openai/whisper-large-v3-turbo \
  --language it
```