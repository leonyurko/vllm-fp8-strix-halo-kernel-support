#!/usr/bin/env bash
# INT8 W8A8 decode sweep: for each public model, A/B stock-Triton-int8 vs our
# rows-GEMV (FP8_GEMV_DECODE=0/1), record mean tok/s, then DELETE the model.
# Stops after TARGET successful models. Runs detached (nohup-safe).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=docker.io/kyuz0/vllm-therock-gfx1151:stable
RES="$HERE/int8_sweep_results.csv"
LOG="$HERE/int8_sweep.log"
TARGET=9
echo "model,baseline_tok_s,ours_tok_s,speedup,kernel" > "$RES"
: > "$LOG"

CANDIDATES=(
  RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8
  RedHatAI/Qwen2.5-1.5B-Instruct-quantized.w8a8
  RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a8
  RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8
  RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8
  RedHatAI/Mistral-7B-Instruct-v0.3-quantized.w8a8
  RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8
  RedHatAI/Qwen2.5-14B-Instruct-quantized.w8a8
  RedHatAI/Qwen2.5-32B-Instruct-quantized.w8a8
  RedHatAI/Qwen2.5-Coder-7B-Instruct-quantized.w8a8
  RedHatAI/gemma-2-9b-it-quantized.w8a8
  RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8
)

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
dl(){ docker run --rm -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        --entrypoint python3 "$IMG" -c "from huggingface_hub import snapshot_download as s; s('$1')"; }
wait_up(){ for i in $(seq 1 72); do sleep 5;
    docker logs vllm_int8 2>&1 | grep -q "Application startup complete" && return 0;
    docker ps --format '{{.Names}}' | grep -q vllm_int8 || return 1; done; return 1; }
purge(){ # files are root-owned (downloaded inside the container), so delete via a root container
  local d="models--$(echo "$1" | sed 's#/#--#g')"
  docker run --rm -v "$HOME/.cache/huggingface":/hf --entrypoint bash "$IMG" -lc "rm -rf /hf/hub/$d" 2>/dev/null
}

OK=0
for M in "${CANDIDATES[@]}"; do
  [ "$OK" -ge "$TARGET" ] && break
  log "===== $M ====="
  if ! dl "$M" >>"$LOG" 2>&1; then log "  SKIP: download/404 failed"; purge "$M"; continue; fi
  # baseline (stock triton int8)
  FP8_GEMV_DECODE=0 bash "$HERE/serve_int8.sh" "$M" 0.85 >/dev/null 2>&1
  if ! wait_up; then log "  SKIP: baseline startup failed"; docker logs vllm_int8 2>&1 | grep -iE 'Selected|Error|fault' | tail -3 >>"$LOG"; docker rm -f vllm_int8 >/dev/null 2>&1; purge "$M"; continue; fi
  KSEL=$(docker logs vllm_int8 2>&1 | grep -oE 'Selected [A-Za-z0-9_]+' | tail -1 | awk '{print $2}')
  B=$(MODEL="$M" VLLM_URL=http://localhost:8102/v1/chat/completions python3 "$HERE/bench_sweep.py")
  docker rm -f vllm_int8 >/dev/null 2>&1
  # ours (rows-GEMV)
  FP8_GEMV_DECODE=1 bash "$HERE/serve_int8.sh" "$M" 0.85 >/dev/null 2>&1
  if ! wait_up; then log "  SKIP: ours startup failed"; docker rm -f vllm_int8 >/dev/null 2>&1; purge "$M"; continue; fi
  O=$(MODEL="$M" VLLM_URL=http://localhost:8102/v1/chat/completions python3 "$HERE/bench_sweep.py")
  docker rm -f vllm_int8 >/dev/null 2>&1
  SP=$(python3 -c "b=$B; o=$O; print(f'{o/b:.2f}' if b>0 else '0')" 2>/dev/null || echo "?")
  echo "$M,$B,$O,$SP,$KSEL" >> "$RES"
  log "  RESULT  baseline=$B  ours=$O  speedup=${SP}x  kernel=$KSEL"
  purge "$M"
  OK=$((OK+1))
  log "  ($OK/$TARGET done)"
done
log "SWEEP DONE — $OK models. Results: $RES"
