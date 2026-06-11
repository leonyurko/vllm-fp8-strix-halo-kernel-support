import torch
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    run_nvfp4_emulations, ref_nvfp4_quant,
)
from nvfp4_triton import nvfp4_gemv
dev = "cuda"; torch.manual_seed(0)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import kE2M1ToFloat_handle
kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(dev)
print("=== NVFP4 fused decode-GEMV correctness vs run_nvfp4_emulations (M=1) ===")
for N, K in [(512, 2048), (1024, 4096), (768, 3072)]:
    x = (torch.randn(1, K, device=dev, dtype=torch.bfloat16) * 0.1)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    ws_fp8 = (torch.rand(N, K // 16, device=dev) * 0.4 + 0.1).to(torch.float8_e4m3fn)
    wgs = torch.tensor(1.0 / 6.0, device=dev, dtype=torch.float32)
    igs = torch.tensor(1.0 / 6.0, device=dev, dtype=torch.float32)
    # reference (materializes full bf16 weight + matmul)
    ref = run_nvfp4_emulations(x, igs, w, ws_fp8, wgs, swizzle=False).to(torch.float32)
    # ours: caller-side activation round-trip (matches emulation) + fused weight GEMV
    x_fp4, x_bs = ref_nvfp4_quant(x, igs, 16)
    x_fp4 = x_fp4.reshape(1, K // 16, 16); x_bs = x_bs.unsqueeze(-1) / igs
    x_dq = (x_fp4 * x_bs).reshape(1, K).to(torch.float32)
    wscale_f32 = ws_fp8.to(torch.float32) * wgs
    out = nvfp4_gemv(x_dq, w, wscale_f32).to(torch.float32)
    rel = ((out - ref).abs().mean() / (ref.abs().mean() + 1e-9)).item()
    print(f"N={N:>5} K={K:>5}  mean_rel={rel:.5f}  {'OK' if rel < 0.02 else 'FAIL'}")
