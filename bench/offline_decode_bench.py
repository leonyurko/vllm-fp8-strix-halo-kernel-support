"""Offline single-process decode bench — wrap with rocprofv3 for a per-kernel
profile (the GPU-activity path of torch.profiler page-faults on this build):

  VLLM_ENABLE_V1_MULTIPROCESSING=0 MODEL=your-org/Your-Model-FP8 \
  rocprofv3 --kernel-trace --hip-trace --stats --output-format csv -d /tmp/prof \
    -- python3 offline_decode_bench.py
"""
import os, time
from vllm import LLM, SamplingParams
MODEL = os.environ.get("MODEL", "your-org/Your-Model-FP8")

def main():
    llm = LLM(model=MODEL, gpu_memory_utilization=0.85, max_model_len=4096,
              enforce_eager=True, dtype="bfloat16")
    llm.generate(["hi"], SamplingParams(max_tokens=8, temperature=0))  # warm
    sp = SamplingParams(max_tokens=32, temperature=0)
    t = time.time()
    out = llm.generate(["Explain how GPU memory bandwidth limits token generation."], sp)
    el = time.time() - t
    ct = len(out[0].outputs[0].token_ids)
    print(f"DECODE_BENCH: {ct} tokens in {el:.2f}s = {ct/el:.2f} tok/s")

if __name__ == "__main__":
    main()
