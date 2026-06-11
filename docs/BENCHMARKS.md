# Benchmarks

## W8A8-INT8 decode: rows-GEMV vs vLLM's stock Triton int8 kernel

Public compressed-tensors **W8A8-INT8** models on one **AMD Strix Halo gfx1151**
iGPU, vLLM `0.19.2rc1`, `--enforce-eager`, decode (batch 1). Each number is the
mean tok/s over 3 runs of a 256-token generation. "baseline" = vLLM's stock
`TritonInt8ScaledMMLinearKernel`; "ours" = the same path with the rows-mapped
dequant-GEMV (`FP8_GEMV_DECODE=1`). Reproduce with
[`bench/orchestrate_int8_sweep.sh`](../bench/orchestrate_int8_sweep.sh); raw data
in [`bench/results/int8_sweep.csv`](../bench/results/int8_sweep.csv).

| model | baseline tok/s | ours tok/s | speedup |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 144.6 | 142.5 | 0.99× |
| Llama-3.2-1B-Instruct | 61.4 | 115.0 | **1.87×** |
| Qwen2.5-1.5B-Instruct | 84.7 | 100.8 | 1.19× |
| Qwen2.5-3B-Instruct | 34.1 | 54.6 | 1.60× |
| Llama-3.2-3B-Instruct | 29.3 | 51.6 | 1.76× |
| Mistral-7B-Instruct-v0.3 | 11.2 | 29.2 | **2.61×** |
| Qwen2.5-7B-Instruct | 25.4 | 29.2 | 1.15× |
| Qwen2.5-14B-Instruct | 10.2 | 15.1 | 1.48× |
| Qwen2.5-32B-Instruct | 4.84 | 6.98 | 1.44× |

**Faster on 8 of 9 models, up to 2.61×, geometric-mean ≈ 1.50×.**

### Reading the numbers honestly

- **The 0.5B is a wash (0.99×).** At that size the per-output K is tiny, decode
  is launch/overhead-bound rather than DRAM-page-bound, so there's nothing for the
  rows-mapping to fix. Expected.
- **The win is shape-dependent, not strictly size-dependent.** Mistral-7B gets
  2.61× while Qwen2.5-7B gets 1.15× — same parameter count, different FFN
  intermediate dims (→ different `K`), so the stock kernel's DRAM page-locality
  penalty differs. Note our kernel pulls *both* 7B models to ~29 tok/s while the
  stock kernel swings 11–25: ours is **more uniform** because it removes the
  layout pathology instead of getting lucky with shapes.
- **One generic config.** All "ours" numbers use a single `BLOCK_N=8, num_warps=8`
  config (no per-model tuning). The lower-speedup rows (e.g. Qwen-7B 1.15×) are
  exactly where a per-shape autotune-at-load (see [ROADMAP](ROADMAP.md)) should
  recover more — this table is the *floor*, not the ceiling.

### Why

Decode reads every weight once per token, so it's memory-bandwidth-bound. The
stock kernel maps each GPU program across many output columns that live far apart
in the transposed `[K,N]` weight, so for large `K` it opens many distant DRAM
pages per step (a row-activation storm). The rows-mapped GEMV instead streams long
contiguous-`K` bursts for a few outputs at a time — far fewer open pages, the bus
streams. See [`EXPLAINER.md`](EXPLAINER.md) §6 for the full analysis and the FP8
results.

> Setup note: numbers come from a single developer box that also runs other GPU
> services, `--enforce-eager`, no tensor-parallel. Treat them as *relative*
> (ours vs stock on identical setup), not as absolute peak throughput.

## Methodology

**Metric.** Decode throughput — tokens generated per second at **batch size 1**.
This is the single-user, interactive regime, and it's the memory-bandwidth-bound
phase this kernel targets. (Prefill is compute-bound and uses a different code
path — see below.)

**The A/B is single-variable.** Baseline and "ours" are the *same* model weights,
container, vLLM build, prompt, machine, and session — with exactly **one** thing
changed: the `FP8_GEMV_DECODE` environment flag.
- `FP8_GEMV_DECODE=0` → the M==1 (decode) matmul falls through to vLLM's stock
  `TritonInt8ScaledMMLinearKernel` (`triton_scaled_mm`).
- `FP8_GEMV_DECODE=1` → that same call routes to the rows-mapped dequant-GEMV.

Nothing else differs, so any delta is attributable to the kernel alone — no
confound from a different model, runtime, or thermal drift between far-apart runs.

**Per data point.**
1. One warm-up generation (so first-call Triton compilation / cache effects don't
   pollute timing).
2. **3 measured runs**, each a 256-token generation from a fixed neutral prompt
   ("write ~220 words about GPU memory bandwidth…"). 256 tokens is long enough that
   per-request setup is negligible and you measure steady-state decode.
3. Report the **mean** tok/s of the 3 runs.

**Per model.** download → serve baseline → bench → tear down → serve ours → bench
→ tear down → append one CSV row → **delete the model** → next. One model resident
at a time (the box has ~96 GB GPU memory but limited system RAM and co-hosts other
services, so the footprint is kept bounded). Fully scripted in
[`bench/orchestrate_int8_sweep.sh`](../bench/orchestrate_int8_sweep.sh) +
[`bench/bench_sweep.py`](../bench/bench_sweep.py).

**Model selection.** Public compressed-tensors **W8A8-INT8** checkpoints
(RedHatAI), deliberately spanning **0.5B → 32B** across **three families**
(Qwen2.5, Llama-3.2, Mistral), to show whether the result holds across scale and
architecture rather than on one lucky shape.

**Averaging.** Speedups are summarized with the **geometric mean** (the correct
average for ratios), not the arithmetic mean.

**What this does not control / claim.**
- **Relative, not absolute** — "ours vs stock on identical setup," not peak
  throughput. `--enforce-eager` (no CUDA-graph capture), batch 1, no
  tensor-parallel, shared box. A clean dedicated box raises *both* columns; the
  ratio is the portable result.
- **One generic kernel config** (`BLOCK_N=8, num_warps=8`) for every model — no
  per-model tuning. The low-speedup rows are where autotune-at-load (see
  [ROADMAP](ROADMAP.md)) should recover more, so this table is a floor.
- **Symmetric int8 only** — the path that routes to our kernel; asymmetric quant
  falls back to stock.
- **n = 3** runs per point (mean reported) — enough to smooth jitter, not a formal
  variance study.

**FP8 vs INT8 framing.** The FP8 numbers in [`EXPLAINER.md`](EXPLAINER.md) are a
*progression of our own implementations* on one 24B model, because vLLM cannot run
FP8 on this hardware at all (no honest stock baseline). The INT8 table here is the
cleaner comparison: vLLM ships a working int8 kernel, so it's a true ours-vs-theirs
A/B on identical setup.
