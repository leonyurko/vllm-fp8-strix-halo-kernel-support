#!/usr/bin/env bash
# Serve a compressed-tensors W8A8-FP8 model in vLLM on an AMD Strix Halo
# (gfx1151 / RDNA3.5) iGPU at ~9 tok/s decode (Q8-comparable) at the FP8 footprint
# — on hardware with NO FP8 compute units.
#
# Stack (all bind-mounted over the stock image, no rebuild):
#   fp8_triton.py        rows-mapped Triton dequant-GEMV (the speed)
#   pytorch_patched.py   routes vLLM's scaled_mm -> fp8_triton (no FP8 hardware)
#   rocm_patched.py      HIP memory-reporting fix: max(vram_total, gtt_total)
# Kernel config: BLOCK_N=8, num_warps=8. See docs/EXPLAINER.md for the full story.
#
# Usage:  ./serve_fast.sh <hf-fp8-model-id> [gpu_mem_util]
#   e.g.  ./serve_fast.sh your-org/Your-Model-FP8 0.85
set -e
IMG=docker.io/kyuz0/vllm-therock-gfx1151:stable
F=/opt/venv/lib64/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/pytorch.py
RP=/opt/venv/lib64/python3.12/site-packages/vllm/platforms/rocm.py
MODEL=${1:?pass a compressed-tensors W8A8-FP8 model id, e.g. ./serve_fast.sh your-org/Your-Model-FP8}
UTIL=${2:-0.85}
HERE=$(cd "$(dirname "$0")" && pwd)
docker rm -f vllm_fp8 2>/dev/null || true
docker run -d --name vllm_fp8 --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  --shm-size 8g -p 8101:8000 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HERE/pytorch_patched.py":"$F":ro \
  -v "$HERE/rocm_patched.py":"$RP":ro \
  -v "$HERE":/opt/fp8:ro \
  -e PYTHONPATH=/opt/fp8 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TRITON_CACHE_DIR=/tmp/triton_cache \
  -e VLLM_ROCM_USE_AITER=0 \
  --entrypoint bash "$IMG" \
  -c "vllm serve $MODEL --host 0.0.0.0 --port 8000 --gpu-memory-utilization $UTIL --max-model-len 8192 --enforce-eager"
echo "serving $MODEL on :8101 util=$UTIL  (docker logs -f vllm_fp8)"
