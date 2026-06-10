import torch, time, triton
import fp8_triton as ft
dev="cuda"
SHAPES=[(5120,6144),(4096,5120),(5120,65536),(32768,5120)]  # qkv,o,gate_up,down

def run(K,N,BN,BK,nw):
    W=(torch.randn(N,K,device=dev)*0.1).to(torch.float8_e4m3fn)
    B=W.t()  # [K,N] stride (1,K) -- real layout
    A=(torch.randn(1,K,device=dev)*0.1).to(torch.float8_e4m3fn)
    SA=torch.ones(1,device=dev); SB=(torch.rand(N,device=dev)*0.5+0.5)
    C=torch.empty((1,N),device=dev,dtype=torch.float32)
    grid=(triton.cdiv(N,BN),)
    def call():
        ft._fp8_gemv_rows[grid](A,B,C,SA,SB,A,N,K,A.stride(1),B.stride(0),B.stride(1),
                                HAS_BIAS=False,BLOCK_N=BN,BLOCK_K=BK,num_warps=nw)
    for _ in range(5): call()
    torch.cuda.synchronize(); it=40; t=time.time()
    for _ in range(it): call()
    torch.cuda.synchronize(); dt=(time.time()-t)/it
    return K*N/dt/1e9, dt

def bestbk(K):
    return max(b for b in (4096,2048,1024,512) if K%b==0)

for K,N in SHAPES:
    BK=bestbk(K); row=[]
    for BN in (4,8,16,32):
        for nw in (4,8):
            try:
                gb,dt=run(K,N,BN,BK,nw); row.append((gb,BN,nw))
            except Exception as e:
                row.append((0,BN,nw))
    row.sort(reverse=True)
    best=row[0]
    print(f"K={K:>6} N={N:>6} BK={BK}: best {best[0]:.0f} GB/s @ BN={best[1]} nw={best[2]}  | "
          + " ".join(f"{g:.0f}(BN{b}/w{w})" for g,b,w in row[:4]))
