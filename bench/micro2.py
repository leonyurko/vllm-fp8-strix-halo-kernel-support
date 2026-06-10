import torch, time
from fp8_triton import fp8_gemm
# (K,N) = (in_features, out_features) for each of the 4 per-layer matmuls
SHAPES = [(5120,6144),(4096,5120),(5120,65536),(32768,5120)]  # qkv, o, gate_up, down
dev="cuda"

def bench(make_B, label):
    tot_t=0.0; tot_b=0
    print(f"--- {label} ---")
    for K,N in SHAPES:
        A=(torch.randn(1,K,device=dev)*0.1).to(torch.float8_e4m3fn)
        B=make_B(K,N)
        sa=torch.ones(1,device=dev); sb=(torch.rand(N,device=dev)*0.5+0.5)
        for _ in range(5): fp8_gemm(A,B,sa,sb,torch.bfloat16,None)
        torch.cuda.synchronize()
        it=30; t=time.time()
        for _ in range(it): fp8_gemm(A,B,sa,sb,torch.bfloat16,None)
        torch.cuda.synchronize(); dt=(time.time()-t)/it
        gbps=K*N/dt/1e9
        print(f"  K={K:>6} N={N:>6} stride={tuple(B.stride())}: {dt*1e6:8.0f} us  {gbps:6.0f} GB/s")
        tot_t+=dt; tot_b+=K*N
    print(f"  AGG {tot_b/tot_t/1e9:.0f} GB/s | layer {tot_t*1e6:.0f}us | est tok/s x40 = {1/(tot_t*40):.2f}\n")

# A: contiguous (K,N) — what micro.py used (the FAST layout)
bench(lambda K,N: (torch.randn(K,N,device=dev)*0.1).to(torch.float8_e4m3fn),
      "CONTIGUOUS (K,N)  [micro.py original]")
# B: strided .t() view of an [N,K] weight — what vLLM actually passes
bench(lambda K,N: (torch.randn(N,K,device=dev)*0.1).to(torch.float8_e4m3fn).t(),
      "STRIDED .t() of [N,K]  [real vLLM layout]")
