"""Experiment only, not for upstream: dump the inputs of a few MORI EPv2 fused_moe calls.

A single-GPU benchmark (gapmoe_bench.py in the handoff) replays them with the received
rows packed densely, as today, and spread over per-source segments with gaps, to
measure what AITER pays for the gapped layout.

Enabled only when the JSON config file exists (default /prof/gapmoe/dump.json; override
with SGLANG_EXP_GAPMOE_DUMP_CFG), for example:
    {"skip": 400, "every": 37, "count": 3, "min_rows": 16384, "ranks": [0]}
"""

import json
import os
import time

import torch

_CFG_PATH = os.environ.get("SGLANG_EXP_GAPMOE_DUMP_CFG", "/prof/gapmoe/dump.json")
_state = {"cfg": None, "seen": 0, "dumped": 0}


def _load_cfg() -> dict:
    if _state["cfg"] is None:
        try:
            with open(_CFG_PATH) as f:
                _state["cfg"] = json.load(f)
        except (OSError, ValueError):
            _state["cfg"] = {}
    return _state["cfg"]


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _to_saveable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    # aiter's QuantType / ActivationType are pybind enums; store them by name.
    if (type(value).__module__ or "").startswith("aiter") and hasattr(value, "name"):
        return {"__aiter_enum__": type(value).__name__, "name": value.name}
    return value


def maybe_dump(kwargs: dict) -> None:
    cfg = _load_cfg()
    if not cfg:
        return
    num_local_tokens = kwargs.get("num_local_tokens")
    hidden_states = kwargs["hidden_states"]
    # The MORI path passes num_local_tokens; the rows are already cut to the step's
    # receive bound, so min_rows picks prefill-sized calls without a host sync.
    if num_local_tokens is None or hidden_states.shape[0] < cfg.get("min_rows", 16384):
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    rank = _rank()
    if rank not in cfg.get("ranks", [0]):
        return
    index = _state["seen"]
    _state["seen"] += 1
    skip, every = cfg.get("skip", 400), cfg.get("every", 37)
    if index < skip or (index - skip) % every or _state["dumped"] >= cfg.get("count", 3):
        return

    out_dir = cfg.get("dir", os.path.dirname(_CFG_PATH))
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    record = {"kwargs": {}, "output_spec": None}
    for key, value in kwargs.items():
        if key == "output":
            if value is not None:
                record["output_spec"] = {"shape": list(value.shape), "dtype": value.dtype}
            continue
        record["kwargs"][key] = _to_saveable(value)
    n_valid = int(num_local_tokens.reshape(-1)[0].item())
    record["meta"] = {
        "rank": rank,
        "call_index": index,
        "rows": hidden_states.shape[0],
        "n_valid": n_valid,
        "torch": torch.__version__,
    }
    path = os.path.join(out_dir, f"gapmoe_r{rank}_c{index}.pt")
    torch.save(record, path)
    _state["dumped"] += 1
    print(
        f"[gapmoe] dumped {path} rows={hidden_states.shape[0]} n_valid={n_valid} "
        f"in {time.time() - start:.1f}s",
        flush=True,
    )
