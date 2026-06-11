import json, time, os, urllib.request
U = os.environ.get("VLLM_URL", "http://localhost:8102/v1/chat/completions")
M = os.environ["MODEL"]
def chat(p, mt):
    b = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0}
    r = urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); d = json.loads(urllib.request.urlopen(r, timeout=600).read()); el = time.time() - t
    return d["usage"]["completion_tokens"], el
try:
    chat("hi", 8)  # warm
    rates = []
    for _ in range(3):
        ct, el = chat("Write a 220-word paragraph about GPU memory bandwidth and how it limits LLM token generation.", 256)
        if el > 0 and ct > 0:
            rates.append(ct / el)
    print(f"{sum(rates)/len(rates):.2f}" if rates else "0")
except Exception:
    print("0")
