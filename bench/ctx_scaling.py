"""Decode rate vs context length — proves decode is flat in context (i.e. the
bottleneck is per-step work / weight reads, not attention/KV). MODEL via env."""
import json, time, os, urllib.request
URL=os.environ.get("VLLM_URL","http://localhost:8101/v1/chat/completions")
MODEL=os.environ.get("MODEL","your-org/Your-Model-FP8")
def call(prompt, mt):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0.0}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time(); d=json.loads(urllib.request.urlopen(req,timeout=900).read()); el=time.time()-t
    u=d["usage"]; return u["prompt_tokens"], u["completion_tokens"], el

def decode_rate(prompt, k=40):
    pt, _, t1 = call(prompt, 1)        # ~ prefill + 1 step
    pt2, ct, tk = call(prompt, k+1)    # prefill + (k+1) steps
    dec = (tk - t1) / k                # per-decode-token seconds at ctx≈pt
    return pt, dec

filler = "Memory bandwidth bounds autoregressive decode on this hardware. "
short = "Hi."
mid   = "Summarize. " + (filler * 60)    # ~ 800 ctx
long  = "Summarize. " + (filler * 240)   # ~ 3200 ctx
for name, p in [("SHORT", short), ("MID", mid), ("LONG", long)]:
    ctx, dec = decode_rate(p)
    print(f"{name:6} ctx~{ctx:>5} tok : {dec*1000:7.1f} ms/decode-token  ->  {1/dec:5.2f} tok/s")
