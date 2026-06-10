# Running FP8 LLMs at Q8 speed on an AMD Strix Halo iGPU (gfx1151) — how, and the math

> We got a 24B FP8 LLM serving in vLLM at **9.24 tokens/s** on an AMD Ryzen AI Max+
> "Strix Halo" iGPU (gfx1151 / RDNA 3.5) — a chip with **no FP8 tensor hardware**,
> which vLLM normally refuses to run FP8 on at all. That's ~the same speed as a Q8
> GGUF in Ollama, but at the FP8 24 GB footprint.
>
> This is the full, ground-up writeup: the hardware reality, every bug, and the
> arithmetic behind each number.

---

## 0. The setup, in one paragraph

Strix Halo is an APU: CPU + RDNA 3.5 iGPU sharing one pool of LPDDR5X over a
~256 GB/s bus. You can hand most of the RAM to the GPU (we run 96 GB as "VRAM").
It's the cheapest way to hold a 24 GB model on "GPU memory" — but it has **no FP8
matrix units** (those start at CDNA3/MI300 and Hopper). So FP8 here is purely a
*memory-footprint* play: store weights at 1 byte/param, dequantize to bf16 on the
fly, do the math in bf16. The question was whether the on-the-fly dequant could
keep up with the memory bus. It can.

Model under test: a 24B-parameter dense decoder, ~40 layers, hidden 5120, standard
(non-MoE) FFN, quantized to compressed-tensors **W8A8-FP8**.
Runtime: vLLM 0.19.2rc1 on the community `kyuz0/vllm-therock-gfx1151` image (the
only ROCm build that even sees this iGPU — stock `rocm/vllm` hangs).

---

## 1. Why decode is *memory-bandwidth-bound* (the single most important idea)

LLM inference has two phases:

- **Prefill**: process the whole prompt at once. A `T`-token prompt multiplies each
  weight by a `T×K` block — lots of arithmetic per weight byte read → **compute-bound**.
- **Decode**: generate one token at a time. Each new token multiplies a **single**
  `1×K` activation vector against every weight. You read all the weights, do almost
  no arithmetic per byte, throw them away, repeat → **memory-bandwidth-bound**.

The decode ceiling is therefore pure division:

```
tokens/sec ≈ memory_bandwidth / bytes_read_per_token
           (bytes_read_per_token ≈ total weight bytes — you read every weight once per token)
```

For this model: **~23 GB of FP8 weights**, **~256 GB/s** bus:

```
time/token ≈ 23 GB / 256 GB/s ≈ 90 ms   →   ceiling ≈ 11 tokens/s
```

Cross-check: the same model as a **Q8 GGUF in Ollama** (also ~24 GB, well-tuned
dequant kernels) does **~10 tok/s** on the same box. So **~10 tok/s is the real,
achievable target** — anything far below means *we* are leaving bandwidth on the
table, not the hardware.

This reframes everything: **FP8 is not slower than bf16 at decode — it's ~2× faster**,
because you read 23 GB/token instead of 46 GB/token. The whole game is making the
dequant-GEMV actually hit bus bandwidth.

---

## 2. The starting point: vLLM won't even run FP8

vLLM's FP8 path calls `torch._scaled_mm`, which errors on RDNA:
`torch._scaled_mm is only supported on ROCm MI300+`. No FP8 units to target. Step
one: **replace that call** with our own kernel.

We bind-mount a patched
`…/vllm/model_executor/kernels/linear/scaled_mm/pytorch.py` (`pytorch_patched.py`)
that routes every FP8 matmul to a custom Triton kernel. Bind-mount, not monkeypatch
— vLLM runs the model in a worker subprocess, so a runtime patch in the parent
never reaches it.

First working version: a naive "upcast to bf16, matmul, apply scales" fallback.
Correct output, **~1.3 tok/s** — 8× below ceiling. The grind begins.

---

## 3. Bug #1 — the 200 ms-per-call transpose copy

vLLM stores a linear weight as `[N, K]` (out × in) and asks for `A · Wᵀ`. The
wrapper was doing `fp8_gemm(A.contiguous(), B.contiguous(), ...)`. `B.contiguous()`
looks innocent, but `B` is a **transposed view** of the weight; `.contiguous()`
physically **copies and re-lays-out** the whole tensor — **every call**:

```
kernel itself:            1,937 µs
the B.contiguous() copy: 199,756 µs   ← 100× the actual math
```

Fix: **don't copy** — pass the strided view, read it via `B.stride()` in the
kernel. Kernel-level cost 201,819 → 3,676 µs (**55×**); end-to-end → **2.08 tok/s**.

> A hidden `O(weights)` copy per call is catastrophic on a bandwidth-bound workload
> and invisible in the Python — you only see it in a microbench.

---

## 4. The detour: a memory-reporting bug that mimicked a hardware problem

vLLM refused to start, claiming only **15.49 GB** of VRAM, though `nvtop`,
`rocminfo`, and Linux `sysfs` all reported the full **96 GB**. Two stacked,
documented Strix Halo bugs:

1. **Kernel bug (≤ 6.15):** older kernels expose only ~15.5 GB to HIP. *Not us* —
   we're on 6.17.
2. **HIP runtime bug (ROCm/hip#3892):** `hipMemGetInfo()` reports the wrong *total*
   on APUs. Allocations beyond 15.5 GB **succeed**; only the reporting API lies. But
   vLLM sizes its KV budget off that bogus total and bails.

The image already ships a sysfs workaround — but it reads **`mem_info_gtt_total`**
only. That's right for the *common* layout (small VRAM carve-out + large GTT). Our
BIOS does the opposite — 96 GB as *dedicated VRAM* — so `gtt_total` is the *small*
~15.5 GB number and the stock patch actively breaks us. Fix: read **both** apertures
and take the max:

```python
real_ceiling = max(mem_info_vram_total, mem_info_gtt_total)   # 96 GB either way
```

plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (UMA-backed allocations fail
without it). `torch.cuda.mem_get_info()` → 88 GB; the model loads. No BIOS change,
no kernel param, no sudo. (This is the fix in `rocm_patched.py` / the proposed
upstream patch.)

> "free memory > total memory" is a known APU signature. Trust `rocminfo`/sysfs over
> the HIP runtime, and check which aperture holds your pool.

---

## 5. The real wall: a profile that killed two hypotheses

At 2.08 tok/s, a microbench of the GEMV said **179 GB/s** (~70% of peak) → *should*
be ~8 tok/s. Where were the other 6 going? Ruled out with measurements:

- **CUDA graphs** (kill launch overhead): 51 decode graphs captured, ran **2.07 vs
  2.08**. → not launch overhead.
- **Prefill**: **205 tok/s**. → compute path healthy; only per-token decode slow.
- **Decode rate vs context** (98 / 881 / 3221 tokens): flat. → not attention/KV.
- **`rocprofv3 --stats`** per-kernel profile (the GPU-activity path of
  `torch.profiler` page-faults on this build; rocprofv3 as an external wrapper does
  not): **92% of decode was in our own GEMV kernels.** `hipMemcpyWithStream`,
  attention, RMSNorm and the FP8 activation-quant kernels were each <2%.

So not memcpy, not the un-fused activation quant (the arithmetic agreed: ~160 tiny
kernels × ~50 µs ≈ 8 ms/token, nowhere near the ~350 ms gap). **It was the GEMV** —
but only *in the model*, not in the microbench.

---

## 6. The root cause: DRAM page locality, not coalescing

The microbench built its weight as a fresh **contiguous** `(K, N)` tensor. vLLM
passes the **transposed view** of the `[N, K]` weight — shape `[K, N]`, **stride
`(1, K)`**. Same kernel, two layouts:

| matmul (K, N)         | contiguous `(K,N)` | **strided `(1,K)` — real vLLM** |
|-----------------------|-------------------:|--------------------------------:|
| qkv (5120, 6144)      | 156 GB/s           | 249 GB/s                        |
| o (4096, 5120)        | 187 GB/s           | 199 GB/s                        |
| gate_up (5120, 65536) | 172 GB/s           | **89 GB/s**                     |
| **down (32768, 5120)**| 202 GB/s           | **22 GB/s** 💀                  |
| **aggregate**         | **180 → 8.1 tok/s**| **48 GB/s → 2.17 tok/s**        |

That **2.17 matches real serving (2.08).** Found it.

**Why `down_proj` collapses to 22 GB/s.** The kernel mapped each program to
`BLOCK_N = 512` **output columns** and looped over `K`. With the weight `[N, K]`
row-major (each output `n` owns a *contiguous* K-long row), 512 outputs at once means
touching **512 different rows simultaneously** — for `down_proj`, `K = 32768`, so
those rows are **32 KB apart**. Every step the controller opens 512 unrelated DRAM
pages. Not a coalescing problem (the reads *are* unit-stride) — a **DRAM
row-activation storm**. The stride math predicts it: `gate_up` (K=5120 → 5 KB apart)
suffers less (89); `down_proj` (32 KB apart) most (22).

**The fix:** invert the aspect ratio. Map each program to **few output rows**
(`BLOCK_N = 8`) and stream **long contiguous K bursts** (`BLOCK_K` up to 4096),
reducing over the contiguous K. Now each program walks a handful of rows mostly
sequentially — few open pages, long bursts, the bus streams. Drop split-K and its
finalize kernel (8 rows/program already yields thousands of programs — plenty for 40
CUs), fold the per-channel weight scale and per-token activation scale into the
epilogue.

Real layout after the fix: `down_proj` **22 → 113 GB/s**, aggregate **48 → 175
GB/s**, end-to-end **2.08 → 6.94 tok/s**, no extra memory. Correctness unchanged
(`mean_rel = 0.0000`).

> A microbench that doesn't replicate the exact tensor strides of the real workload
> will lie. On bandwidth-bound kernels, **how you walk memory across the parallel
> dimension (page locality)** matters more than raw coalescing.

---

## 7. The last 33%: occupancy

The config sweep showed `num_warps = 4 → 8` is a ~1.5× win on the big shapes (more
waves in flight to hide memory latency). `BLOCK_N = 4`/`16` measured even faster in
isolation but **page-faulted at vLLM warmup** (a Triton tile-shape miscompile on
this iGPU), so the shipped config is the stable `BLOCK_N = 8, num_warps = 8`.

End-to-end: **9.24 tok/s** (256 tokens in 27.7 s, reproducible, correct).

---

## 8. The scoreboard

| stage | tok/s | the fix | the math |
|---|---:|---|---|
| bf16-upcast fallback | 1.3 | naive correctness | 8× below ceiling |
| drop `B.contiguous()` | 2.08 | kill 200 ms/call copy | 55× kernel-level |
| rows-mapped GEMV | 6.94 | DRAM page locality | down_proj 22→113 GB/s |
| `num_warps = 8` | **9.24** | occupancy / latency hiding | ~1.5× on big shapes |

**9.24 tok/s ≈ Ollama Q8 (~10), at the FP8 24 GB footprint, on an iGPU with no FP8
hardware.** Decode is now within ~15% of the memory-bandwidth ceiling.

---

## 9. What generalizes

- The GEMV is **layout-bound, not dtype-bound** — the same structure ports to
  **W8A8-INT8** (swap the FP8 cast for an INT8 dequant) and to any 8-bit weight-only
  decode on a bandwidth-bound device. *(WIP — see ROADMAP.)*
- Per-shape tuning is **model-specific** (different `K,N`), so the clean form is an
  **autotune-at-load** that runs the sweep once for the model's shapes. *(WIP.)*
- The HIP-memory `max(vram,gtt)` fix is a small, upstreamable patch.

## 10. Reproduce

Box: gfx1151, Linux ≥ 6.16, `kyuz0/vllm-therock-gfx1151:stable`. `serve_fast.sh`
bind-mounts `fp8_triton.py` (the rows-mapped dequant-GEMV), `pytorch_patched.py`
(routes `scaled_mm` → our kernel), `rocm_patched.py` (the `max(vram,gtt)` fix), with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `VLLM_ROCM_USE_AITER=0`.
Kernel config `BLOCK_N=8, num_warps=8, BLOCK_K=largest pow2 dividing K`. Profile with
`rocprofv3 --kernel-trace --hip-trace --stats` on `bench/offline_decode_bench.py`
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`).
