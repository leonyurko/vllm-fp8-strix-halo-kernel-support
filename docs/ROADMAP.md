# Roadmap

## Done

- **FP8 W8A8** — 9.24 tok/s (rows-GEMV + HIP mem fix).
- **W8A8-INT8** — up to 2.61× vs stock (geomean 1.5×, 9 models). Same rows-GEMV.
- **NVFP4** — 0.42 → 7.13 tok/s = 17× (fused FP4 dequant-GEMV vs the bf16-
  materializing emulation). Numerically validated.

## 🚧 NVFP4 floor-chase (work in progress)

NVFP4 is at 7.13 tok/s but the kernel ceiling (~8.3) is still ~13% of the ~64 FP4
floor — FP4 reads half the bytes of INT8, so this is the highest-ceiling path.
- Profile the FP4 GEMV (rocprofv3) to find the structural wall (same playbook that
  took W8A8 from 2→9), then sweep `BLOCK_N`/`BLOCK_K`/`num_warps`.
- Fuse the activation FP4 round-trip into the kernel (currently caller-side; it's
  only ~13% of the cost, so secondary).

## 🚧 More formats

- **AWQ-int4** — `awq_marlin` runs at ~54% of floor on gfx1151 → ~1.8× opportunity
  (moderate); separate path to patch.
- **act-order W4A16** — only `ConchLinearKernel` serves act-order (g_idx) models on
  gfx1151, at ~23% of floor. A validated 4-bit rows-GEMV exists (`w4a16_triton.py`)
  but the non-act-order TritonW4A16 path is already near floor; the win is the Conch
  path, which needs **g_idx activation-reordering + Conch's uint8 layout**.
- Mature INT4 (GPTQ/Marlin) are already near floor — not worth porting.

## 🚧 Generic quantization-format support / autotune-at-load (work in progress)

- The per-`(K,N)` `BLOCK_N`/`num_warps` config is **model-shape-specific**. Replace
  the hardcoded map with a **one-time autotune at load** that sweeps the model's
  actual matmul shapes and caches the winners — makes the repo a true drop-in for
  any model / any 8-bit format.
- Broaden coverage across compressed-tensors variants (per-tensor vs per-channel
  vs per-token scales) and validate on multiple model architectures.

## Last-mile kernel ideas

- Recover the `BLOCK_N=4`/`16` bandwidth (faster in isolation) without the gfx1151
  Triton warmup page-fault — newer Triton, or guard the tile shapes.
- `uint8`-aliased `dwordx4` vectorized loads + in-register bitcast for the big-K
  shape (`down_proj`).
- Pre-transpose-at-load alternative (`process_weights_after_loading`,
  `.t().contiguous().t()` — no memory doubling) if a wide-N kernel is ever wanted.

## Validation / hardening

- Test across more models and context lengths; track quality (not just speed).
- Upstream the `max(vram_total, gtt_total)` HIP memory fix.
