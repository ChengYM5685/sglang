"""Logical receive-view bounds shared by the MORI EPv1 and EPv2 dispatchers.

The validated intranode MORI kernels (EPv1 IntraNode/AsyncLL, EPv2 FlyDSL/HIP)
send at most one dense row per ``(source token, destination rank)``, so the sum
of the sender input rows bounds every rank's receive count independent of the
router top-k. The bound is only used when DP scheduler metadata maps one-to-one
onto the EP senders and matches the current local dispatch; every unproved case
keeps the full physical view.
"""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple, Optional, Sequence

MORI_MIN_LOGICAL_RECV_ROWS = 32
MORI_LOGICAL_RECV_ROW_ALIGN = 32


class MoriRecvBoundDecision(NamedTuple):
    rows: int
    reason: str
    sender_rows: tuple[int, ...] | None


def normalize_sender_rows(
    sender_rows: Optional[Sequence[int]],
) -> tuple[int, ...] | None:
    """Copy CPU scheduler metadata without introducing a device sync."""
    if not isinstance(sender_rows, (list, tuple)):
        return None
    if any(
        isinstance(rows, bool) or not isinstance(rows, Integral) for rows in sender_rows
    ):
        return None
    return tuple(int(rows) for rows in sender_rows)


def round_logical_recv_rows(cluster_rows: int, *, pow2_buckets: bool) -> int:
    # Measured on MI355X with DSV4 FP4 AITER fused MoE: latency tracks the logical
    # M and a new M costs no JIT, so power-of-two padding only pays off for a
    # backend that pre-builds power-of-two receive caps.
    if pow2_buckets:
        return max(MORI_MIN_LOGICAL_RECV_ROWS, 1 << (cluster_rows - 1).bit_length())
    align = MORI_LOGICAL_RECV_ROW_ALIGN
    return max(MORI_MIN_LOGICAL_RECV_ROWS, -(-cluster_rows // align) * align)


def mori_recv_bound_decision(
    *,
    enabled: bool,
    physical_rows: int,
    local_rows: int,
    sender_rows: Optional[Sequence[int]],
    ep_size: int,
    ep_rank: int,
    tp_size: int,
    attn_dp_size: int,
    attn_dp_rank: int,
    moe_tp_size: int,
    moe_dp_size: int,
    tbo_enabled: bool,
    layout_verified: bool,
    pow2_buckets: bool,
    explicit_cluster_rows: Optional[int] = None,
) -> MoriRecvBoundDecision:
    full = max(0, int(physical_rows))
    if not enabled:
        return MoriRecvBoundDecision(full, "disabled", None)
    if physical_rows <= 0:
        return MoriRecvBoundDecision(full, "invalid_capacity", None)
    if tbo_enabled:
        return MoriRecvBoundDecision(full, "tbo_metadata_missing", None)
    if not layout_verified:
        return MoriRecvBoundDecision(full, "layout_unverified", None)
    if (
        ep_size <= 1
        or tp_size != ep_size
        or attn_dp_size != ep_size
        or moe_tp_size != 1
        or moe_dp_size != 1
        or not 0 <= ep_rank < ep_size
        or attn_dp_rank != ep_rank
    ):
        return MoriRecvBoundDecision(full, "sender_mapping_unknown", None)

    normalized = normalize_sender_rows(sender_rows)
    if normalized is None:
        return MoriRecvBoundDecision(full, "metadata_missing", None)
    if len(normalized) != ep_size or any(rows < 0 for rows in normalized):
        return MoriRecvBoundDecision(full, "metadata_invalid", normalized)
    if local_rows < 0 or normalized[ep_rank] != local_rows:
        return MoriRecvBoundDecision(full, "metadata_mismatch", normalized)

    cluster_rows = sum(normalized)
    if explicit_cluster_rows is not None:
        if (
            isinstance(explicit_cluster_rows, bool)
            or not isinstance(explicit_cluster_rows, Integral)
            or int(explicit_cluster_rows) != cluster_rows
        ):
            return MoriRecvBoundDecision(full, "metadata_mismatch", normalized)

    # A globally empty MoE dispatch is not a production path we currently
    # optimize. Keeping the full view avoids introducing a new empty fast path.
    if cluster_rows <= 0:
        return MoriRecvBoundDecision(full, "no_saving", normalized)
    if cluster_rows > physical_rows:
        return MoriRecvBoundDecision(full, "capacity_unproven", normalized)

    rows = round_logical_recv_rows(cluster_rows, pow2_buckets=pow2_buckets)
    if rows >= physical_rows:
        return MoriRecvBoundDecision(full, "no_saving", normalized)
    return MoriRecvBoundDecision(rows, "trimmed_dedup", normalized)
