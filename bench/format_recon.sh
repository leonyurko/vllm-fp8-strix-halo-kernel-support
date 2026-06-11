#!/usr/bin/env bash
# Format-support recon on gfx1151: for each sub-16-bit format, record
#   loads? | quant method | selected kernel | stock decode tok/s | weight GB | floor | %floor
# Stock kernels only (mem-fix mounted, no patches). One model at a time, deleted after.
# Self-queues: waits until no vllm_* container holds the GPU before starting.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=docker.io/kyuz0/vllm-therock-gfx1151:stable
RP=/opt/venv/lib64/python3.12/site-packages/vllm/platforms/rocm.py
RES="$HERE/format_recon_results.csv"
LOG="$HERE/format_recon.log"
echo "format,model,loaded,quant,kernel,stock_tok_s,weight_gb,floor_tok_s,pct_floor" > "$RES"
: > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# format|model
CAND=(
  "awq_int4|Qwen/Qwen2.5-7B-Instruct-AWQ"
  "gptq_int4|Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
  "ct_w4a16|RedHatAI/Qwen2.5-7B-Instruct-quantized.w4a16"
  "bnb_nf4|unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  "nvfp4|RedHatAI/Qwen3-8B-NVFP4"
  "nvfp4b|nvidia/Llama-3.1-8B-Instruct-FP4"
  "mxfp4|openai/gpt-oss-20b"
)

dl(){ docker run --rm -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        --entrypoint python3 "$IMG" -c "from huggingface_hub import snapshot_download as s; s('$1')"; }
purge(){ docker run --rm -v "$HOME/.cache/huggingface":/hf --entrypoint bash "$IMG" \
        -lc "rm -rf /hf/hub/models--$(echo "$1" | sed 's#/#--#g')" 2>/dev/null; }
wait_up(){ for i in $(seq 1 72); do sleep 5;
  docker logs vllm_fmt 2>&1 | grep -q "Application startup complete" && return 0;
  docker ps --format '{{.Names}}' | grep -q vllm_fmt || return 1; done; return 1; }

# --- wait for the GPU to free (previous recon container gone) ---
log "waiting for GPU to free…"
for i in $(seq 1 240); do docker ps --format '{{.Names}}' | grep -qE 'vllm_recon|vllm_int8|vllm_fp8' || break; sleep 15; done
log "GPU free — starting format recon"

for entry in "${CAND[@]}"; do
  FMT="${entry%%|*}"; M="${entry#*|}"
  log "===== $FMT : $M ====="
  if ! dl "$M" >>"$LOG" 2>&1; then log "  NOTFOUND/404"; echo "$FMT,$M,no(404),,,,,," >>"$RES"; continue; fi
  WB=$(docker run --rm -v "$HOME/.cache/huggingface":/hf --entrypoint bash "$IMG" -lc "du -sb /hf/hub/models--$(echo "$M"|sed 's#/#--#g')/snapshots 2>/dev/null | cut -f1")
  GB=$(python3 -c "print(f'{$WB/1e9:.1f}')" 2>/dev/null || echo 0)
  docker rm -f vllm_fmt 2>/dev/null || true
  docker run -d --name vllm_fmt --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render --security-opt seccomp=unconfined --shm-size 8g -p 8104:8000 \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v "$HERE/rocm_patched.py":"$RP":ro \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_ROCM_USE_AITER=0 \
    --entrypoint bash "$IMG" \
    -c "vllm serve $M --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.7 --max-model-len 4096 --enforce-eager" >/dev/null 2>&1
  if ! wait_up; then
    ERR=$(docker logs vllm_fmt 2>&1 | grep -iE "Error|not support|no kernel|assert|NotImplemented|fault" | tail -1 | cut -c1-80)
    log "  NO-RUN on gfx1151: $ERR"
    echo "$FMT,$M,no-run,,,,$GB,," >>"$RES"
    docker rm -f vllm_fmt >/dev/null 2>&1; purge "$M"; continue
  fi
  QUANT=$(docker logs vllm_fmt 2>&1 | grep -oE "quantization=[a-z0-9_]+" | tail -1 | cut -d= -f2)
  KSEL=$(docker logs vllm_fmt 2>&1 | grep -oE "Selected [A-Za-z0-9_]+|[A-Za-z]*Marlin[A-Za-z]*|GPTQ[A-Za-z]*Method|AWQ[A-Za-z]*Method" | tail -1)
  TOKS=$(MODEL="$M" VLLM_URL=http://localhost:8104/v1/chat/completions python3 "$HERE/bench_sweep.py")
  FLOOR=$(python3 -c "g=$GB; print(f'{256/g:.1f}' if g>0 else '0')" 2>/dev/null || echo 0)
  PCT=$(python3 -c "t=$TOKS; f=256/$GB if $GB>0 else 0; print(f'{100*t/f:.0f}%' if f>0 else '?')" 2>/dev/null || echo "?")
  echo "$FMT,$M,yes,$QUANT,$KSEL,$TOKS,$GB,$FLOOR,$PCT" >>"$RES"
  log "  loaded  quant=$QUANT  kernel=$KSEL  stock=$TOKS tok/s  ${GB}GB  floor~$FLOOR  ${PCT} of floor"
  docker rm -f vllm_fmt >/dev/null 2>&1
  purge "$M"
done
log "FORMAT RECON DONE — $RES"
