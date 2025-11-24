#!/bin/bash

MODEL=meta-llama/Llama-3.1-8B-Instruct
MAX_MODEL_LEN=131072

HIP_VERBOSE=0 \
CUDA_VISIBLE_DEVICES=0 \
    python server_hf.py \
      --model $MODEL \
      --host 0.0.0.0 \
      --port 8082 \
      --attn-implementation window \
      --mode delta \
      --sliding-window 2048 \
      --delta-lambda 64 \
