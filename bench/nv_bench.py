import json, time, os, urllib.request
U = os.environ.get("VLLM_URL", "http://localhost:8107/v1/chat/completions")
M = os.environ["MODEL"]
def chat(p, mt):
    b = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0}
    r = urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); d = json.loads(urllib.request.urlopen(r, timeout=900).read()); el = time.time() - t
    return d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"], el
chat("hi", 4)
txt, ct, el = chat("Explain GPU memory bandwidth in 40 words.", 64)
ok = len(txt.strip()) > 20
print(f"tok_s={ct/el:.2f} tokens={ct} elapsed={el:.1f}s coherent={ok}")
