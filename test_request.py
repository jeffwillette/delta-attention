#!/usr/bin/env python3
import argparse, json, time, requests

def load_prompt():
    # with open("test_input_32k.txt", "r") as f:
    #     out = json.load(f)
    with open("test_input.txt", "r") as f:
        out = json.load(f)

    return out["input"], out["outputs"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--approx-tokens", type=int, default=12000, help="Approximate prompt token count (default ~12k). Increase to 20k/32k+ if you want.")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    # Keep top_k=0 to avoid topk out-of-range issues
    ap.add_argument("--method", choices=["PUT","POST"], default="PUT")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/generate"
    prompt, output = load_prompt()

    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,     # 0.0 => greedy
            "top_p": args.top_p,
            "top_k": 0,                          # disable top-k to avoid index errors
            "stop": []                           # add stop strings if you like
        },
        "stream": False
    }

    t0 = time.time()
    method = args.method.upper()
    print(f"[client] sending request...")
    if method == "PUT":
        r = requests.put(url, json=payload, timeout=600)
    else:
        r = requests.post(url, json=payload, timeout=600)
    dt = time.time() - t0

    try:
        r.raise_for_status()
    except Exception as e:
        print("[client] HTTP error:", e)
        print("[client] Response text:", r.text[:2000])
        return

    # Server returns {"text": "..."} for single prompt
    resp = r.json()
    print(resp)
    text = resp.get("text") if isinstance(resp, dict) else resp[0]["text"]
    print(f"[client] status={r.status_code} time={dt:.2f}s, output_chars={len(text):,}")
    print("----- MODEL OUTPUT (first 500 chars) -----")
    print(text[:500])
    print("----- EXPECTED OUTPUT -----")
    print(output)

if __name__ == "__main__":
    main()
