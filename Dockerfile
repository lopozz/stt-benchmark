FROM vllm/vllm-openai:latest
RUN pip install --no-cache-dir "vllm[audio]"