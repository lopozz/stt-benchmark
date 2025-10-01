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

# Run Benchmark
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