from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import torch

import sglang.srt.layers.moe.token_dispatcher.moriepv2 as adapter
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.moriepv2 import (
    MoriEPv2Dispatcher,
    _mori_epv2_recv_bound_decision,
    _resolve_tbo_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_tbo_geometry_defaults_are_phase_specific():
    assert _resolve_tbo_geometry(
        tbo_enabled=True,
        dispatch_block_num=32,
        combine_block_num=48,
        dispatch_warp_num_per_block=4,
        combine_warp_num_per_block=4,
    ) == (32, 4, 48, 4)


def test_non_tbo_geometry_preserves_tuned_schedule():
    assert _resolve_tbo_geometry(
        tbo_enabled=False,
        dispatch_block_num=-1,
        combine_block_num=-1,
        dispatch_warp_num_per_block=-1,
        combine_warp_num_per_block=-1,
    ) == (None, None, None, None)


@pytest.mark.parametrize("field", range(4))
def test_tbo_geometry_rejects_non_positive_values(field):
    values = [32, 48, 4, 4]
    values[field] = 0
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_tbo_geometry(
            tbo_enabled=True,
            dispatch_block_num=values[0],
            combine_block_num=values[1],
            dispatch_warp_num_per_block=values[2],
            combine_warp_num_per_block=values[3],
        )


def _dispatcher_for_quant_test():
    dispatcher = MoriEPv2Dispatcher.__new__(MoriEPv2Dispatcher)
    BaseDispatcher.__init__(dispatcher)
    dispatcher.dispatch_dtype = torch.bfloat16
    dispatcher.fp4_quant_func = object()
    dispatcher._op = None
    dispatcher._initialize_op = Mock()
    return dispatcher


def test_quant_config_selects_fp4_asymmetric_transport(monkeypatch):
    monkeypatch.delenv("SGLANG_MORI_EPV2_DISPATCH_DTYPE", raising=False)
    dispatcher = _dispatcher_for_quant_test()
    dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
    assert dispatcher.dispatch_dtype == torch.float4_e2m1fn_x2
    dispatcher._initialize_op.assert_called_once_with()


def test_quant_config_defaults_to_bf16(monkeypatch):
    monkeypatch.delenv("SGLANG_MORI_EPV2_DISPATCH_DTYPE", raising=False)
    dispatcher = _dispatcher_for_quant_test()
    dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert dispatcher.dispatch_dtype == torch.bfloat16


def test_fp4_override_and_invalid_override(monkeypatch):
    dispatcher = _dispatcher_for_quant_test()
    monkeypatch.setenv("SGLANG_MORI_EPV2_DISPATCH_DTYPE", "fp4")
    dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert dispatcher.dispatch_dtype == torch.float4_e2m1fn_x2

    dispatcher = _dispatcher_for_quant_test()
    monkeypatch.setenv("SGLANG_MORI_EPV2_DISPATCH_DTYPE", "invalid")
    with pytest.raises(ValueError, match="must be bf16 or fp4"):
        dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("comm_stream", [False, True])
def test_recv_capacity_api_compatibility(monkeypatch, dynamic, comm_stream):
    op = SimpleNamespace(
        cfg=SimpleNamespace(effective_max_recv=64),
        dispatch=Mock(return_value=(None, None, None, None, None, object())),
    )
    if dynamic:
        op.prepare_recv_cap = Mock()
    monkeypatch.setattr(adapter, "init_mori_epv2_op", Mock(return_value=op))
    monkeypatch.setattr(adapter, "get_int_env_var", lambda name, default: default)
    monkeypatch.setattr(torch, "cuda", MagicMock())
    dispatcher = Mock()
    MoriEPv2Dispatcher._initialize_op(dispatcher)
    if dynamic:
        assert op.prepare_recv_cap.call_args_list == [call(32), call(64)]
    dispatcher.op = op
    dispatcher._trim_recv = False
    dispatcher._direct_output = False
    dispatcher._select_recv_cap.return_value = 32
    dispatcher._comm_stream = Mock() if comm_stream else None
    dispatcher._dispatch_intermediate_state = (None,) * 5 + (Mock(),)
    MoriEPv2Dispatcher.dispatch_b(dispatcher)
    kwargs = {"return_routing": True}
    if dynamic:
        kwargs.update(recv_cap=32, clone_routing=False)
    op.dispatch.assert_called_once_with(None, None, None, None, **kwargs)


@pytest.mark.parametrize(
    "sender_rows,rank,expected,reason",
    [
        ([1] * 8, 0, 32, "trimmed_dedup"),
        ([7] * 8, 0, 64, "trimmed_dedup"),
        ([56] * 8, 3, 512, "trimmed_dedup"),
        ([448] * 8, 7, 4096, "trimmed_dedup"),
        ([0, 1, 7, 33, 56, 128, 257, 448], 3, 1024, "trimmed_dedup"),
        ([0, 1, 7, 33, 56, 128, 257, 448], 0, 1024, "trimmed_dedup"),
        ([8192] * 8, 0, 65536, "no_saving"),
        ([0] * 8, 0, 65536, "no_saving"),
    ],
)
def test_epv2_deduplicated_bound(sender_rows, rank, expected, reason):
    decision = _mori_epv2_recv_bound_decision(
        enabled=True,
        physical_rows=65536,
        local_rows=sender_rows[rank],
        sender_rows=sender_rows,
        ep_size=8,
        ep_rank=rank,
        tp_size=8,
        attn_dp_size=8,
        attn_dp_rank=rank,
        moe_tp_size=1,
        moe_dp_size=1,
        tbo_enabled=False,
        kernel_backend="flydsl",
        is_internode=False,
    )
    assert decision.rows == expected
    assert decision.reason == reason


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"enabled": False}, "disabled"),
        ({"sender_rows": None}, "metadata_missing"),
        ({"sender_rows": [7] * 7}, "metadata_invalid"),
        ({"sender_rows": [7] * 7 + [-1]}, "metadata_invalid"),
        ({"local_rows": 8}, "metadata_mismatch"),
        ({"explicit_cluster_rows": 57}, "metadata_mismatch"),
        ({"tp_size": 16}, "sender_mapping_unknown"),
        ({"attn_dp_size": 4}, "sender_mapping_unknown"),
        ({"attn_dp_rank": 1}, "sender_mapping_unknown"),
        ({"moe_tp_size": 2}, "sender_mapping_unknown"),
        ({"moe_dp_size": 2}, "sender_mapping_unknown"),
        ({"tbo_enabled": True}, "tbo_metadata_missing"),
        ({"kernel_backend": "unknown"}, "layout_unverified"),
        ({"is_internode": True}, "layout_unverified"),
        ({"kernel_backend": "hip", "is_internode": True}, "layout_unverified"),
        ({"sender_rows": [9000] * 8, "local_rows": 9000}, "capacity_unproven"),
    ],
)
def test_epv2_bound_falls_back_when_safety_is_unproved(overrides, reason):
    kwargs = {
        "enabled": True,
        "physical_rows": 65536,
        "local_rows": 7,
        "sender_rows": [7] * 8,
        "ep_size": 8,
        "ep_rank": 0,
        "tp_size": 8,
        "attn_dp_size": 8,
        "attn_dp_rank": 0,
        "moe_tp_size": 1,
        "moe_dp_size": 1,
        "tbo_enabled": False,
        "kernel_backend": "flydsl",
        "is_internode": False,
        "explicit_cluster_rows": None,
    }
    kwargs.update(overrides)
    decision = _mori_epv2_recv_bound_decision(**kwargs)
    assert decision.rows == 65536
    assert decision.reason == reason


def test_epv2_hip_intranode_deduplicated_bound():
    decision = _mori_epv2_recv_bound_decision(
        enabled=True,
        physical_rows=65536,
        local_rows=56,
        sender_rows=[56] * 8,
        ep_size=8,
        ep_rank=0,
        tp_size=8,
        attn_dp_size=8,
        attn_dp_rank=0,
        moe_tp_size=1,
        moe_dp_size=1,
        tbo_enabled=False,
        kernel_backend="hip",
        is_internode=False,
    )
    assert decision.rows == 512
    assert decision.reason == "trimmed_dedup"


@pytest.mark.parametrize("ep_size,expected", [(2, 128), (4, 256)])
def test_epv2_deduplicated_bound_for_smaller_ep_groups(ep_size, expected):
    decision = _mori_epv2_recv_bound_decision(
        enabled=True,
        physical_rows=ep_size * 8192,
        local_rows=56,
        sender_rows=[56] * ep_size,
        ep_size=ep_size,
        ep_rank=0,
        tp_size=ep_size,
        attn_dp_size=ep_size,
        attn_dp_rank=0,
        moe_tp_size=1,
        moe_dp_size=1,
        tbo_enabled=False,
        kernel_backend="flydsl",
        is_internode=False,
    )
    assert decision.rows == expected
    assert decision.reason == "trimmed_dedup"


def test_select_recv_cap_uses_current_nonuniform_sender_snapshot(monkeypatch):
    sender_rows = [0, 1, 7, 33, 56, 128, 257, 448]
    rank = 3
    monkeypatch.setattr(
        adapter,
        "get_parallel",
        lambda: SimpleNamespace(
            moe_ep_size=8,
            moe_ep_rank=rank,
            tp_size=8,
            attn_dp_size=8,
            attn_dp_rank=rank,
            moe_tp_size=1,
            moe_dp_size=1,
            launch_world_rank=rank,
        ),
    )
    dispatcher = SimpleNamespace(
        op=SimpleNamespace(
            cfg=SimpleNamespace(effective_max_recv=65536, is_internode=False),
            backend_name="flydsl",
        ),
        _trim_recv=True,
        _tbo_enabled=False,
        _num_tokens=sender_rows[rank],
    )
    assert (
        MoriEPv2Dispatcher._select_recv_cap(
            dispatcher,
            explicit_cluster_rows=sum(sender_rows),
            sender_rows=sender_rows,
        )
        == 1024
    )
    assert dispatcher._recv_bound_reason == "trimmed_dedup"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
