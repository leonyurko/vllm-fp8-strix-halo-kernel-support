#!/usr/bin/env python3
"""Throughput + correctness harness for the FP8 vLLM endpoint on :8101.

  MODEL=your-org/Your-Model-FP8 python3 bench.py
"""
import json, sys, time, os, urllib.request
URL = os.environ.get("VLLM_URL", "http://localhost:8101/v1/chat/completions")
MODEL = os.environ.get("MODEL", "your-org/Your-Model-FP8")
def chat(prompt, max_tokens=256, temperature=0.0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    el = time.time() - t
    return d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"], el
def main():
    chat("hi", max_tokens=8)  # warm
    txt, _, _ = chat("17+25=? Reply with only the number.", max_tokens=64)
    ok_math = "42" in txt
    _, ct, el = chat("Write a 200-word paragraph about memory bandwidth in GPUs.", max_tokens=256)
    tps = ct / el if el else 0
    print(f"correctness_math={ok_math}  completion_tokens={ct}  elapsed={el:.1f}s  tok_s={tps:.2f}")
    print("SAMPLE:", txt[:160].replace(chr(10), " "))
    sys.exit(0 if ok_math else 1)
if __name__ == "__main__":
    main()
