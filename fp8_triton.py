import torch, triton, triton.language as tl

# ---- Decode GEMV (M==1) for the layout vLLM actually passes:
# B = [K,N] strided (1,K) == .t() of the [N,K] row-major weight, so every output
# n owns a CONTIGUOUS K-run. The win is program mapping, not coalescing: map
# programs over OUTPUT ROWS (few rows, long contiguous K bursts) so the DRAM
# controller sees few open pages. Reduce over the contiguous k. No split-K, no
# k-mask (BLOCK_K chosen to divide K), scales folded into the epilogue.
# cudagraph-safe: no autotune, output buffer reused per (N) across replays.
_GEMV_BN = 8
_GEMV_WARPS = 4
# (K,N) -> (BLOCK_N, num_warps). BN=8 is the stable tile on gfx1151 (BN=4/16
# can trigger a Triton page-fault at warmup); num_warps=8 is a safe ~1.5x win
# over nw=4 on this hardware (sweep_gemv.py). BLOCK_K from _pick_bk.
_GEMV_CFG = {}
_GEMV_DEFAULT = (8, 8)

_BUF = {}
def _get_out(N, device):
    key = (N, device.index if device.index is not None else -1)
    c = _BUF.get(key)
    if c is None:
        c = torch.empty((1, N), device=device, dtype=torch.float32)
        _BUF[key] = c
    return c

@triton.jit
def _fp8_gemv_rows(A, B, C, SA, SB, BIAS, N, K, sak, sbk, sbn,
                   HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)
        a = tl.load(A + offs_k * sak).to(tl.bfloat16)                       # K-vector (L2 resident)
        b = tl.load(B + offs_n[:, None] * sbn + offs_k[None, :] * sbk,      # [BLOCK_N, BLOCK_K], k innermost & unit-stride
                    mask=nmask[:, None], other=0.0).to(tl.bfloat16)
        acc += tl.sum((b * a[None, :]).to(tl.float32), axis=1)              # reduce over contiguous k
    sa0 = tl.load(SA)
    sb = tl.load(SB + offs_n, mask=nmask, other=1.0)
    acc = acc * sa0 * sb
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=nmask, other=0.0)
    tl.store(C + offs_n, acc, mask=nmask)

@triton.jit
def _fp8_gemm(A, B, C, SA, SB, BIAS, M, N, K, sam, sak, sbk, sbn, scm, scn,
              HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None]*sam + offs_k[None, :]*sak
    b_ptrs = B + offs_k[:, None]*sbk + offs_n[None, :]*sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k0)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k0)) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), out_dtype=tl.float32)
        a_ptrs += BLOCK_K*sak; b_ptrs += BLOCK_K*sbk
    sa = tl.load(SA + offs_m, mask=offs_m < M, other=1.0)
    sb = tl.load(SB + offs_n, mask=offs_n < N, other=1.0)
    acc = acc * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    c_ptrs = C + offs_m[:, None]*scm + offs_n[None, :]*scn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def _pick_bk(K):
    for bk in (4096, 2048, 1024, 512, 256, 128, 64):
        if K % bk == 0:
            return bk
    return 64

def fp8_gemm(A, B, scale_a, scale_b, out_dtype, bias=None):
    assert A.is_contiguous()  # B is a strided (transposed) weight view; kernel uses B.stride()
    M, K = A.shape; K2, N = B.shape; assert K == K2
    sa = scale_a.reshape(-1).to(torch.float32); sb = scale_b.reshape(-1).to(torch.float32)
    if sa.numel() == 1: sa = sa.expand(M).contiguous()
    if sb.numel() == 1: sb = sb.expand(N).contiguous()
    if M == 1:
        C = _get_out(N, A.device)
        BLOCK_K = _pick_bk(K)
        BN, NW = _GEMV_CFG.get((K, N), _GEMV_DEFAULT)
        grid = (triton.cdiv(N, BN),)
        _fp8_gemv_rows[grid](A, B, C, sa, sb, bias if bias is not None else A, N, K,
                             A.stride(1), B.stride(0), B.stride(1),
                             HAS_BIAS=bias is not None, BLOCK_N=BN, BLOCK_K=BLOCK_K,
                             num_warps=NW)
        return C.to(out_dtype)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    bias_t = bias if bias is not None else A
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_gemm[grid](A, B, C, sa, sb, bias_t, M, N, K,
                    A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                    HAS_BIAS=bias is not None, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C.to(out_dtype)
