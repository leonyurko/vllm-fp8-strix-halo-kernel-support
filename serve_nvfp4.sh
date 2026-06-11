#!/usr/bin/env bash
set -e
IMG=docker.io/kyuz0/vllm-therock-gfx1151:stable
NU=/opt/venv/lib64/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
RP=/opt/venv/lib64/python3.12/site-packages/vllm/platforms/rocm.py
MODEL=${1:?pass an NVFP4 model id}; UTIL=${2:-0.6}
HERE=$(cd "$(dirname "$0")" && pwd)
docker rm -f vllm_nv 2>/dev/null || true
docker run -d --name vllm_nv --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined --shm-size 8g -p 8107:8000 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HERE/nvfp4_emulation_utils_patched.py":"$NU":ro \
  -v "$HERE/rocm_patched.py":"$RP":ro \
  -v "$HERE":/opt/fp8:ro \
  -e PYTHONPATH=/opt/fp8 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TRITON_CACHE_DIR=/tmp/triton_cache -e VLLM_ROCM_USE_AITER=0 \
  -e NVFP4_GEMV=${NVFP4_GEMV:-1} \
  -e NVFP4_SKIP_ACT=${NVFP4_SKIP_ACT:-0} \
  --entrypoint bash "$IMG" \
  -c "vllm serve $MODEL --host 0.0.0.0 --port 8000 --gpu-memory-utilization $UTIL --max-model-len 4096 --enforce-eager"
echo "serving NVFP4 $MODEL on :8107 NVFP4_GEMV=${NVFP4_GEMV:-1}"
