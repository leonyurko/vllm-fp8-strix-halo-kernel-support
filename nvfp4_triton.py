import torch, triton, triton.language as tl

# Fused NVFP4 decode-GEMV for M==1 decode. The stock EmulationNvFp4LinearKernel
# materializes the ENTIRE weight to bf16 every forward (run_nvfp4_emulations ->
# dequantize_to_dtype -> torch.matmul) ~ reads 4GB FP4 then writes+reads ~16GB
# bf16 => ~0.4 tok/s. Here we fuse: read FP4 weight [N,K//2] uint8 (row-major, so
# each output's K-row is contiguous) + per-16 block scales, dequant in-register,
# never materialize bf16. Activation round-trip (x_dq) is done by the caller
# (cheap at M=1) to match emulation numerics exactly.
#
# E2M1 decode: nibble -> kE2M1[nibble & 7] * (nibble&8 ? -1 : 1),
#   kE2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6]   (passed as an 8-elem f32 table)
# Byte j packs K-indices [2j (low nibble), 2j+1 (high nibble)].
_NV_BN = 16
_NV_BK = 128   # multiple of 16 (block_size)
_NV_WARPS = 8
_BUF = {}
def _get_out(N, device):
    k = (N, device.index if device.index is not None else -1)
    c = _BUF.get(k)
    if c is None:
        c = torch.empty((1, N), device=device, dtype=torch.float32); _BUF[k] = c
    return c

@triton.jit
def _nvfp4_gemv(x_ptr, w_ptr, ws_ptr, c_ptr, N, K,
                stride_xk, stride_wn, stride_wk, stride_sn, stride_sk,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        x = tl.load(x_ptr + offs_k * stride_xk, mask=kmask, other=0.0).to(tl.float32)  # [BK]
        offs_b = (k0 // 2) + tl.arange(0, BLOCK_K // 2)                                  # byte cols
        bmask = offs_b < (K // 2)
        wb = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_b[None, :] * stride_wk,
                     mask=nmask[:, None] & bmask[None, :], other=0).to(tl.int32)          # [BN, BK//2]
        low = wb & 0xF
        high = (wb >> 4) & 0xF
        nib = tl.interleave(low, high)                                                    # [BN, BK]  (low0,high0,..)
        idx = nib & 0x7
        # arithmetic E2M1 magnitude (no gather): e=idx>>1, m=idx&1
        #   e==0 -> 0.5*m ; else (1 + 0.5*m) * 2^(e-1)   => {0,.5,1,1.5,2,3,4,6}
        e = idx >> 1
        m = (idx & 1).to(tl.float32)
        shamt = tl.where(e == 0, 0, e - 1)
        pe = (1 << shamt).to(tl.float32)
        mag = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * pe)
        sgn = tl.where((nib & 0x8) != 0, -1.0, 1.0)
        val = mag * sgn                                                                   # E2M1 decode [BN, BK]
        # block scale: load ONE per (n, 16-block) -> [BN, BK//16], broadcast to [BN, BK]
        gbase = k0 // 16
        gcols = gbase + tl.arange(0, BLOCK_K // 16)
        ws_blk = tl.load(ws_ptr + offs_n[:, None] * stride_sn + gcols[None, :] * stride_sk,
                         mask=nmask[:, None] & (gcols[None, :] < (K // 16)), other=0.0)    # [BN, BK//16]
        ws = tl.reshape(tl.broadcast_to(ws_blk[:, :, None], (BLOCK_N, BLOCK_K // 16, 16)),
                        (BLOCK_N, BLOCK_K))                                                # [BN, BK]
        w_dq = val * ws
        acc += tl.sum(x[None, :] * w_dq, axis=1)                                          # reduce over k -> [BN]
    tl.store(c_ptr + offs_n, acc, mask=nmask)

_TBL = {}
def _table(device):
    t = _TBL.get(device.index if device.index is not None else -1)
    if t is None:
        t = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device)
        _TBL[device.index if device.index is not None else -1] = t
    return t

def nvfp4_gemv(x_dq, w_packed_u8, wscale_f32):
    # x_dq: [1, K] dequantized activation (caller-prepared). w_packed_u8: [N, K//2] uint8.
    # wscale_f32: [N, K//16] f32 == weight_scale.to(f32) * weight_global_scale.
    M, K = x_dq.shape
    N = w_packed_u8.shape[0]
    C = _get_out(N, x_dq.device)
    grid = (triton.cdiv(N, _NV_BN),)
    _nvfp4_gemv[grid](x_dq, w_packed_u8, wscale_f32, C, N, K,
                      x_dq.stride(1), w_packed_u8.stride(0), w_packed_u8.stride(1),
                      wscale_f32.stride(0), wscale_f32.stride(1),
                      BLOCK_N=_NV_BN, BLOCK_K=_NV_BK, num_warps=_NV_WARPS)
    return C
