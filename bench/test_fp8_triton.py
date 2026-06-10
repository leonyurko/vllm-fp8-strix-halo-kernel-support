"""Correctness for fp8_gemm vs reference, across aligned, ragged, and single-token shapes."""
import torch
from fp8_triton import fp8_gemm
def _ref(A, B, sa, sb, bias):
    o = (A.to(torch.float32) @ B.to(torch.float32)) * sa.reshape(-1,1) * sb.reshape(1,-1)
    return o + bias.to(torch.float32) if bias is not None else o
def check(M, K, N, scalar_scale, use_bias):
    dev="cuda"
    A=(torch.randn(M,K,device=dev)*0.1).to(torch.float8_e4m3fn)
    B=(torch.randn(K,N,device=dev)*0.1).to(torch.float8_e4m3fn)
    if scalar_scale:
        sa=torch.tensor([1.0],device=dev); sb=torch.tensor([1.0],device=dev)
        rsa=torch.ones(M,device=dev); rsb=torch.ones(N,device=dev)
    else:
        sa=torch.rand(M,device=dev)*0.5+0.5; sb=torch.rand(N,device=dev)*0.5+0.5
        rsa=sa; rsb=sb
    bias=torch.randn(N,device=dev) if use_bias else None
    out=fp8_gemm(A,B,sa,sb,torch.bfloat16,bias)
    ref=_ref(A,B,rsa,rsb,bias).to(torch.bfloat16)
    rel=(out.float()-ref.float()).abs().mean()/(ref.float().abs().mean()+1e-6)
    print(f"M={M} K={K} N={N} scalar={scalar_scale} bias={use_bias} mean_rel={rel:.4f}")
    assert rel<0.05, f"deviates rel={rel}"
def main():
    torch.manual_seed(0)
    check(64,512,256,False,True)      # aligned, per-channel, bias
    check(17,500,129,False,True)      # ragged M,K,N
    check(1,5120,13824,True,False)    # single-token decode, large K/N, scalar scale
    check(3,4096,11008,False,False)   # small ragged batch, no bias
    print("OK")
main()
