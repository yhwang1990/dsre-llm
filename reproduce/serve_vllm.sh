#!/bin/bash
# Start an OpenAI-compatible vLLM server for precompute.py (non-thinking mode is set per request; greedy decoding).
# Usage: bash reproduce/serve_vllm.sh <model dir> <port> [gpu memory fraction]
#   Qwen3-8B -> 8000, Qwen3-4B -> 8001, Llama-3-8B-Instruct -> 8002 (the ports assumed by reproduce/lib.sh)
MODEL=$1; PORT=${2:-8000}; FRAC=${3:-0.45}
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$(basename $MODEL)" \
  --port "$PORT" --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization "$FRAC" --enable-prefix-caching \
  --max-num-seqs 128 --guided-decoding-backend xgrammar --disable-log-requests
