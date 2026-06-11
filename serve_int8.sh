#!/usr/bin/env bash
# Serve a compressed-tensors W8A8-INT8 model on gfx1151, routing M==1 decode to
# the rows-mapped dequant-GEMV (same kernel as FP8). For symmetric int8 models.
set -e
IMG=docker.io/kyuz0/vllm-therock-gfx1151:stable
TMM=/opt/venv/lib64/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py
RP=/opt/venv/lib64/python3.12/site-packages/vllm/platforms/rocm.py
MODEL=${1:?pass a W8A8-INT8 model id}
UTIL=${2:-0.85}
HERE=$(cd "$(dirname "$0")" && pwd)
docker rm -f vllm_int8 2>/dev/null || true
docker run -d --name vllm_int8 --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  --shm-size 8g -p 8102:8000 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HERE/triton_scaled_mm_patched.py":"$TMM":ro \
  -v "$HERE/rocm_patched.py":"$RP":ro \
  -v "$HERE":/opt/fp8:ro \
  -e PYTHONPATH=/opt/fp8 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TRITON_CACHE_DIR=/tmp/triton_cache \
  -e VLLM_ROCM_USE_AITER=0 \
  -e FP8_GEMV_DECODE=${FP8_GEMV_DECODE:-1} \
  --entrypoint bash "$IMG" \
  -c "vllm serve $MODEL --host 0.0.0.0 --port 8000 --gpu-memory-utilization $UTIL --max-model-len 8192 --enforce-eager"
echo "serving INT8 $MODEL on :8102 util=$UTIL"
