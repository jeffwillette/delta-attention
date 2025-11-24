import os
from argparse_dataclass import dataclass
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, ScanStage
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum



@dataclass
class Config:
    # model configuration options
    model_str: str = "meta-llama/Llama-3.1-8B-Instruct"

    # training configuration options
    attn_implementation: str = "window"     # [window, hip_attention, flash_attention_2]
    mode: str = "delta"                     # ["delta", "recompute", "sparse-only"]

    hip_attn_args: Optional[HiPAttentionArgs] = None
    port: int = 8080
    host: str = "0.0.0.0"
    trust_remote_code: bool = True

    temperature: float = 0.0
    top_k: int = 1
    top_p: int = 1
    stop_words: str = ""
    max_new_tokens: int = 128
    repetition_penalty: int = 1

    delta_lambda: int = 64          # delta lambda window
    sliding_window: int = 2048      # sliding window size

    # when using the hip_attention interface as the sparse attention method,
    # the first few layers are set as dense as well as a few dense tokens 
    # before starting dense decode.
    hip_attention_dense_layers = [0, 1, 2]
    hip_attention_last_dense = 64


def get_hip_config(config: Config, layer_idx: int):
    preset_name = os.environ.get("PRESET", "default")
    if preset_name == "default":
        stages = [
            ScanStage(
                stage_block_size_q=128,
                stage_block_stride_q=4,
                stage_chunk_size=256,
                stage_k=None,
                stage_stride=1,
                using_landmark=False,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=8,
                stage_k=32768,
                stage_stride=1,
                using_landmark=False,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=2,
                stage_k=8192,
                stage_stride=1,
                using_landmark=False,
            ),
        ]

        args = HiPAttentionArgs(
            sliding_window_size=1024,
            sink_token_size=1024,
            using_extend=False,
            need_apply_rope=False,
            second_stage_k=2048,
            stages=stages,
            model_context_length=131072,
            scan_extend_backend=("streaming" if layer_idx < 3 else "relative"),
            sa_extend_backend="streaming",
            block_sparse_block_size_q=stages[-1].stage_block_size_q,
            rope_range=(0, 128),
            using_landmark=True,
        )
    else:
        raise NotImplementedError(f"{preset_name=} not implemented")

    return args
