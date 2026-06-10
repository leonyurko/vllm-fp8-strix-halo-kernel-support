# Roadmap

The FP8 path is done (9.24 tok/s). These are the next stages.

## 🚧 W8A8-INT8 (work in progress)

The decode bottleneck is **layout / memory-bandwidth-bound, not dtype-bound**, so
the same rows-mapped GEMV applies almost verbatim to INT8:

- Swap the FP8→bf16 cast in the kernel prologue for an INT8 dequant
  (`(b.to(bf16)) * weight_scale`), keeping the rows-mapped traversal and epilogue.
- vLLM routes INT8 through a **different** `ScaledMMLinearKernel` than FP8, so
  `pytorch_patched.py` needs an INT8 entry point patched too.
- Expectation: same ~9 tok/s class (same bytes/token), and INT8 has a potential edge
  via integer `tl.dot` accumulate if it ever helps on this arch.

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
