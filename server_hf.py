#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serve /generate with HF + Delta.

POST /generate
{
  "text": "prompt" | ["p1","p2",...],
  "sampling_params": {
      "max_new_tokens": 128,
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 0,
      "stop": ["</end>"]
  },
  "stream": false
}
Returns {"text": "..."} or [{"text":"..."},...].
"""

import argparse, json, os, sys
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import traceback
from delta_attention.model_wrapper import HuggingFaceModel
from delta_attention.config import Config

# --- HF deps
from argparse_dataclass import ArgumentParser
from transformers.cache_utils import DynamicCache
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList
)

# Optional: presence/frequency penalties (newer Transformers)
try:
    from transformers.generation.logits_process import (
        PresencePenaltyLogitsProcessor, FrequencyPenaltyLogitsProcessor
    )
    _HAS_P_F = True
except Exception:
    _HAS_P_F = False

try:
    from accelerate import infer_auto_device_map  # noqa: F401 (just to detect)
    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False



# ==================== Helpers ====================

class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_ids: List[List[int]]):
        super().__init__()
        self.stop_ids = [torch.tensor(s, dtype=torch.long) for s in stop_ids if len(s)]
        self.maxL = max((len(s) for s in self.stop_ids), default=0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.maxL == 0:
            return False
        cur = input_ids[0] if input_ids.dim() == 2 else input_ids
        if cur.size(0) < self.maxL:
            return False
        tail = cur[-self.maxL:]
        for s in self.stop_ids:
            L = s.size(0)
            if L and L <= tail.size(0):
                if torch.equal(tail[-L:], s):
                    return True
        return False


def _map_dtype(s: str):
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    if s == "auto":
        return "auto"
    # default
    return torch.bfloat16


def _eos_id_list(tok: AutoTokenizer) -> List[int]:
    ids = []
    for t in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        if isinstance(tok.eos_token_id, int):
            ids.append(tok.eos_token_id)
        elif isinstance(tok.eos_token_id, list):
            ids.extend([i for i in tok.eos_token_id if isinstance(i, int)])
    return list(dict.fromkeys(ids))


def _map_sampling(params: Dict[str, Any]) -> Dict[str, Any]:
    max_new = int(params.get("max_new_tokens", 128))
    temperature = float(params.get("temperature", 0.0))
    do_sample = params.get("do_sample")
    if do_sample is None:
        do_sample = temperature > 0.0
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 0))
    rep_pen = float(params.get("repetition_penalty", 1.0))

    return dict(
        max_new_tokens=max_new,
        temperature=temperature if do_sample else 1.0,
        do_sample=bool(do_sample),
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=rep_pen,
    )


def _get_max_ctx(model, tokenizer) -> int:
    # Conservative: prefer model.config.max_position_embeddings if sane; else tokenizer.model_max_length.
    m = getattr(model.config, "max_position_embeddings", None)
    if isinstance(m, int) and m > 0 and m < 10**9:
        return m
    t = getattr(tokenizer, "model_max_length", None)
    if isinstance(t, int) and t > 0 and t < 10**9:
        return t
    # Fallback large number if unknown
    return 2**31 - 1


def _build_stopping(stop_strs: List[str], tokenizer: AutoTokenizer) -> Optional[StoppingCriteriaList]:
    if not stop_strs:
        return None
    stop_ids = []
    for s in stop_strs:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if toks:
            stop_ids.append(toks)
    if not stop_ids:
        return None
    return StoppingCriteriaList([StopOnSequences(stop_ids)])


def _build_logits_processors(pres_pen: float, freq_pen: float):
    lp = []
    if _HAS_P_F:
        if abs(pres_pen) > 1e-8:
            lp.append(PresencePenaltyLogitsProcessor(pres_pen))
        if abs(freq_pen) > 1e-8:
            lp.append(FrequencyPenaltyLogitsProcessor(freq_pen))
    # If not available, silently skip (HF older version)
    return lp if lp else None


# ==================== App / Global State ====================

app = FastAPI(title="HF + Delta /generate", version="1.0")

STATE: Dict[str, Any] = {
    "model": None,      # HF model
    "max_ctx": None,    # int
}


def load_hf(config: Config):

    model = HuggingFaceModel(
        config,
        do_sample=config.temperature > 0,
        repetition_penalty=config.repetition_penalty,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        stop=config.stop_words,
        max_new_tokens=config.max_new_tokens,
    )

    device = torch.device("cuda:0")
    max_ctx = _get_max_ctx(model.model, model.tokenizer)

    STATE.update(model=model, max_ctx=max_ctx)


# ==================== HTTP ====================
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.api_route("/generate", methods=["POST", "PUT"])
async def generate(req: Request):
    # --- parse body robustly (JSON string, dict, list, or form) ---
    try:
        raw = await req.body()
        body = None
        if raw and raw.strip():
            body = json.loads(raw)
        else:
            # Some clients send application/x-www-form-urlencoded with a "data" field
            try:
                form = await req.form()
                payload = form.get("data") or form.get("request")
                if payload:
                    body = json.loads(payload)
            except Exception:
                pass
        if body is None:
            raise ValueError("empty body")
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

    # Allow top-level list as shorthand for {"text": [...]}
    if isinstance(body, list):
        body = {"text": body}
    if not isinstance(body, dict):
        raise HTTPException(400, "Body must be an object")

    # --- normalize fields RULER/OpenAI-like clients might use ---
    text = None
    for key in ("text", "prompt", "prompts", "input", "inputs"):
        if key in body:
            text = body[key]
            break
    if text is None:
        raise HTTPException(400, "Missing 'text'/'prompt'/'prompts'/'input'/'inputs'")

    # Always operate on a list of prompts
    prompts = text if isinstance(text, list) else [text]

    sp = _map_sampling(body.get("sampling_params", {}) or {})

    try:
        texts = STATE["model"](prompts, **sp)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Generation error: {e}")

    return JSONResponse(texts if isinstance(texts, list) else texts)


def main():
    parser = ArgumentParser(Config)
    config = parser.parse_args()

    print(f"running with config: {config=}")

    assert config.mode in ["delta", "recompute", "sparse-only"], f"{config.mode=} is invalid"
    config.attn_implementation_original = config.attn_implementation
    if config.attn_implementation == "flash_attention_2":
        print(f"{config.attn_implementation=}...ignoring mode setting")

    if not _HAS_ACCELERATE:
        print("[warn] accelerate not detected; model will be ignored and model will load on a single device.",
              file=sys.stderr)

    load_hf(config)

    import uvicorn
    uvicorn.run(app, port=config.port, log_level="info")


if __name__ == "__main__":
    main()

