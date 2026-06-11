import torch
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import triton_w4a16_gemm
from w4a16_triton import w4a16_gemv
dev = "cuda"
torch.manual_seed(0)
# (K, N, group_size, has_zp)
CASES = [(4096, 4096, 128, False), (5120, 13824, 128, False),
         (2048, 2048, 64, True), (3584, 4608, 128, False)]
print("=== W4A16 rows-GEMV correctness vs stock triton_w4a16_gemm (M=1) ===")
for K, N, G, has_zp in CASES:
    a = (torch.randn(1, K, device=dev, dtype=torch.float16) * 0.1)
    b_q = torch.randint(-2**31, 2**31 - 1, (K, N // 8), device=dev, dtype=torch.int32)
    scales = (torch.rand(K // G, N, device=dev, dtype=torch.float16) * 0.05 + 0.005)
    qz = torch.randint(-2**31, 2**31 - 1, (K // G, N // 8), device=dev, dtype=torch.int32) if has_zp else None
    ref = triton_w4a16_gemm(a, b_q, scales, qz, G, zp_bias=8)
    out = w4a16_gemv(a, b_q, scales, qz, G, zp_bias=8).to(ref.dtype)
    rel = ((out - ref).abs().mean() / (ref.abs().mean() + 1e-9)).item()
    print(f"K={K:>5} N={N:>5} G={G:>4} has_zp={has_zp}  mean_rel={rel:.5f}  {'OK' if rel < 0.02 else 'FAIL'}")
