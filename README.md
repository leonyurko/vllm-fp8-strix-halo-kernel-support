# vllm-fp8-strix-halo-kernel-support

**Run FP8 LLMs in vLLM at Q8 speed on AMD Strix Halo (gfx1151 / RDNA 3.5) iGPUs —
on hardware with no FP8 tensor units.**

A custom rows-mapped Triton **dequant-GEMV** + a HIP memory-reporting fix take a 24B
FP8 model from "vLLM refuses to run it" to **9.24 tok/s decode** — about the same as
a Q8 GGUF in Ollama, but at the FP8 ~24 GB footprint.

| stage | tok/s | fix |
|---|---:|---|
| bf16-upcast fallback | 1.3 | naive correctness (no FP8 hardware) |
| drop `B.contiguous()` | 2.08 | kill a 200 ms/call transpose copy |
| rows-mapped GEMV | 6.94 | fix DRAM page locality (down_proj 22→113 GB/s) |
| **`num_warps=8`** | **9.24** | occupancy / memory-latency hiding |

Decode is memory-bandwidth-bound: ~23 GB of weights ÷ ~256 GB/s ≈ 90 ms/token ≈ an
**11 tok/s ceiling**. We land within ~15% of it. Full story + the math:
[`docs/EXPLAINER.md`](docs/EXPLAINER.md).

> ⚠️ Status: **FP8 works (9.24 tok/s).** W8A8-INT8 and a generic
> autotune-at-load are **work in progress** — see [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Hardware / software

- AMD Ryzen AI Max+ "Strix Halo" APU, **gfx1151** (RDNA 3.5) iGPU, unified LPDDR5X.
- Linux **kernel ≥ 6.16** (older kernels cap HIP at ~15.5 GB; see EXPLAINER §4).
- vLLM `0.19.2rc1` inside the community image
  [`kyuz0/vllm-therock-gfx1151:stable`](https://hub.docker.com/r/kyuz0/vllm-therock-gfx1151)
  — the only ROCm build that drives this iGPU (stock `rocm/vllm` hangs).
- A **compressed-tensors W8A8-FP8** model (any; bring your own).

## How it works (3 bind-mounted files, no image rebuild)

| file | what it does |
|---|---|
| [`fp8_triton.py`](fp8_triton.py) | the rows-mapped Triton dequant-GEMV (decode) + a tiled GEMM (prefill). **The speed.** |
| [`pytorch_patched.py`](pytorch_patched.py) | replaces vLLM's `torch._scaled_mm` (unsupported on RDNA) → routes to `fp8_triton`. Modified from vLLM. |
| [`rocm_patched.py`](rocm_patched.py) | fixes the APU `hipMemGetInfo` bug: report `max(vram_total, gtt_total)` so vLLM sees the full pool. Modified from vLLM / the kyuz0 image. |

Kernel config: `BLOCK_N=8, num_warps=8, BLOCK_K = largest power-of-two dividing K`.
The decode kernel is cudagraph-safe (no autotune, reused output buffer).

## Quickstart

```bash
git clone https://github.com/leonyurko/vllm-fp8-strix-halo-kernel-support
cd vllm-fp8-strix-halo-kernel-support
# serve any compressed-tensors W8A8-FP8 model (downloaded to ~/.cache/huggingface)
./serve_fast.sh your-org/Your-Model-FP8 0.85          # -> OpenAI API on :8101
# benchmark
MODEL=your-org/Your-Model-FP8 python3 bench/bench.py
```

`serve_fast.sh` runs the stock image with the three files bind-mounted and these env
vars: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (required for UMA-backed
allocations), `VLLM_ROCM_USE_AITER=0` (the AITER attention path coredumps on
gfx1151), `--enforce-eager`.

## Benchmarks & tools (`bench/`)

| script | purpose |
|---|---|
| `bench.py` | end-to-end correctness + tok/s against the `:8101` endpoint |
| `micro2.py` | the layout A/B: GEMV at contiguous `(K,N)` vs strided `(1,K)` |
| `sweep_gemv.py` | per-shape `BLOCK_N`/`num_warps` sweep |
| `offline_decode_bench.py` | offline single-process decode (wrap with `rocprofv3 --stats`) |
| `ctx_scaling.py` | decode rate vs context length (proves it's not attention) |
| `test_fp8_triton.py` | numerical correctness vs reference |

## Tuning for your model

`fp8_triton.py` ships a small per-`(K,N)` config map plus a safe default. The
optimal `BLOCK_N`/`num_warps` is model-shape-specific — run `bench/sweep_gemv.py`
with your model's matmul shapes and update the map. (Auto-tune-at-load is on the
roadmap.)

## Credits

`pytorch_patched.py` and `rocm_patched.py` are modified from
[vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) and the
[`kyuz0/vllm-therock-gfx1151`](https://github.com/kyuz0/amd-strix-halo-toolboxes)
image — see [`NOTICE`](NOTICE).

## License

Source-available, **noncommercial**. The original work in this repository —
`fp8_triton.py`, the scripts under `bench/`, and the docs — is licensed under
[**PolyForm Noncommercial 1.0.0**](LICENSE): free for any noncommercial purpose;
commercial use requires a separate license from the author.

The two vLLM-derived files (`pytorch_patched.py`, `rocm_patched.py`) remain under
the [**Apache License 2.0**](licenses/Apache-2.0.txt) as required by their upstream
license — see [`NOTICE`](NOTICE).
