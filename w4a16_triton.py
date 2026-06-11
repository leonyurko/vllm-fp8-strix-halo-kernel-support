import torch, triton, triton.language as tl

# Decode (M==1) rows-GEMV for W4A16 (GPTQ-packed int4 weights), mirroring the
# stock triton_w4a16 dequant EXACTLY (interleave x3 + shift, (w-zero)*scale) but:
#   * one program per BLOCK_N outputs, reduce over a long contiguous-K burst
#     (fixes the DRAM page-locality / M->BLOCK_M tl.dot padding waste at M=1)
# Layout (post process_weights_after_loading):
#   b_q   : [K, N//8] int32  (8 int4 vals per int32 along N, shifts [0,4,..,28])
#   scales: [K//G, N]
#   qzeros: [K//G, N//8] int32 or None (None -> symmetric uint4b8, zero=ZP_BIAS)
_W4_BN = 32
_W4_WARPS = 8
_BUF = {}
def _get_out(N, device):
    k = (N, device.index if device.index is not None else -1)
    c = _BUF.get(k)
    if c is None:
        c = torch.empty((1, N), device=device, dtype=torch.float32); _BUF[k] = c
    return c

@triton.jit
def _w4a16_gemv(a_ptr, b_ptr, scales_ptr, zeros_ptr, c_ptr, N, K,
                stride_ak, stride_bk, stride_bn, group_size,
                HAS_ZP: tl.constexpr, ZP_BIAS: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = offs_n < N
    bnmask = offs_bn < (N // 8)
    shifts_row = tl.arange(0, 8) * 4
    shifts_1d = tl.reshape(tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8)), (BLOCK_N,))
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        a = tl.load(a_ptr + offs_k * stride_ak, mask=kmask, other=0.0).to(tl.float32)
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_packed = tl.load(b_ptrs, mask=kmask[:, None] & bnmask[None, :], other=0)
        b = tl.interleave(b_packed, b_packed); b = tl.interleave(b, b); b = tl.interleave(b, b)
        shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))
        b = (b >> shifts) & 0xF
        g_idx = (k_start * BLOCK_K) // group_size
        scales = tl.load(scales_ptr + g_idx * N + offs_n, mask=nmask, other=1.0).to(tl.float32)
        if HAS_ZP:
            z_packed = tl.load(zeros_ptr + g_idx * (N // 8) + offs_bn, mask=bnmask, other=0)
            z = tl.interleave(z_packed, z_packed); z = tl.interleave(z, z); z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xF
        else:
            z = tl.full((BLOCK_N,), ZP_BIAS, dtype=tl.int32)
        b_fp = (b - z[None, :]).to(tl.float32) * scales[None, :]   # [BLOCK_K, BLOCK_N]
        acc += tl.sum(a[:, None] * b_fp, axis=0)
    tl.store(c_ptr + offs_n, acc, mask=nmask)

def w4a16_gemv(a, b_q, scales, qzeros, group_size, zp_bias=8):
    M, K = a.shape
    N = b_q.shape[1] * 8
    BK = group_size if (0 < group_size <= 128) else 128
    C = _get_out(N, a.device)
    grid = (triton.cdiv(N, _W4_BN),)
    _w4a16_gemv[grid](a, b_q, scales, qzeros if qzeros is not None else b_q, C, N, K,
                      a.stride(1), b_q.stride(0), b_q.stride(1), group_size,
                      HAS_ZP=qzeros is not None, ZP_BIAS=zp_bias,
                      BLOCK_N=_W4_BN, BLOCK_K=BK, num_warps=_W4_WARPS)
    return C.to(a.dtype)
