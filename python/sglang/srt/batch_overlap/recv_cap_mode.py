"""Experimental runtime switch for the MORI EPv2 MoE receive cap.

Benchmarks can compare cap policies inside one server process instead of
restarting it (every MORI EPv2 process exit currently leaves ownerless VRAM on
the node). The mode is read from a file, re-checked at most every 0.5 s:

  child        (default) each TBO micro-batch is bounded by its own per-rank rows
  parent       each TBO micro-batch is bounded by its parent batch's rows
  child+exact  child, plus the exact received row count in eager prefill
  parent+exact parent, plus the exact received row count in eager prefill
"""

import os
import time

_MODE_FILE = os.environ.get(
    "SGLANG_MORI_RECV_CAP_MODE_FILE", "/tmp/sglang_mori_recv_cap_mode"
)
_cache = {"t": -1.0, "mode": "child"}


def recv_cap_mode() -> str:
    now = time.monotonic()
    if now - _cache["t"] >= 0.5:
        _cache["t"] = now
        try:
            with open(_MODE_FILE) as f:
                _cache["mode"] = f.read().strip() or "child"
        except OSError:
            _cache["mode"] = "child"
    return _cache["mode"]


def use_child_rows() -> bool:
    return not recv_cap_mode().startswith("parent")


def use_exact_cap() -> bool:
    return recv_cap_mode().endswith("+exact")
