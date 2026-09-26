import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="stage-b-test-cpu-intel")


def _runner_input():
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    return AiterRunnerInput(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        topk_ids=topk_ids,
        topk_weights=torch.ones(topk_ids.shape, dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )


def _quant_info(**overrides):
    kwargs = {
        "w13_weight": torch.empty((2, 8, 2)),
        "w2_weight": torch.empty((2, 4, 2)),
        "quant_type": AiterQuantType.PER_1X32,
    }
    kwargs.update(overrides)
    return AiterMoeQuantInfo(**kwargs)


def _install_fake_aiter(monkeypatch, fused_moe):
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_moe_common = ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.GateMode = SimpleNamespace(
        INTERLEAVE=SimpleNamespace(value="INTERLEAVE")
    )

    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_common", fake_moe_common)


def test_aiter_runner_forwards_no_combine_and_extra_fused_moe_kwargs(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )

    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", no_combine=True))

    runner.run(
        _runner_input(),
        _quant_info(fused_moe_kwargs={"custom_fused_moe_kwarg": "enabled"}),
        running_state={},
    )

    assert captured["activation"] == "Silu"
    assert captured["quant_type"] == "per_1x32"
    assert captured["no_combine"] is True
    assert captured["custom_fused_moe_kwarg"] == "enabled"


def test_aiter_runner_rejects_no_combine_when_fused_moe_does_not_support_it(
    monkeypatch,
):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: False
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))

    with pytest.raises(NotImplementedError, match="no_combine=True"):
        runner.run(_runner_input(), _quant_info(), running_state={})


def test_aiter_runner_preserves_no_combine_rank_for_empty_input(monkeypatch):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))
    runner_input = _runner_input()
    runner_input.hidden_states = torch.zeros((0, 4), dtype=torch.bfloat16)
    runner_input.topk_ids = torch.zeros((0, 2), dtype=torch.int32)
    runner_input.topk_weights = torch.zeros((0, 2), dtype=torch.float32)

    output = runner.run(runner_input, _quant_info(), running_state={})

    assert output.hidden_states.shape == (0, 2, 4)


_FAKE_KERNEL_TYPE = SimpleNamespace(
    IntraNode="IntraNode",
    AsyncLL="AsyncLL",
    InterNodeV1="InterNodeV1",
    InterNodeV1LL="InterNodeV1LL",
)


def _patch_epv1_recv_bound(
    monkeypatch, *, sender_rows, rank=0, tbo=False, ep_size=None
):
    import sglang.srt.layers.dp_attention as dp_attention
    import sglang.srt.layers.moe.utils as moe_utils

    fake_mori = ModuleType("mori")
    fake_mori.ops = SimpleNamespace(EpDispatchCombineKernelType=_FAKE_KERNEL_TYPE)
    monkeypatch.setitem(sys.modules, "mori", fake_mori)
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "1")
    monkeypatch.setattr(dp_attention, "get_dp_global_num_tokens", lambda: sender_rows)
    monkeypatch.setattr(moe_utils, "is_tbo_enabled", lambda: tbo)
    ep_size = ep_size or len(sender_rows)
    monkeypatch.setattr(
        aiter_runner,
        "get_parallel",
        lambda: SimpleNamespace(
            moe_ep_size=ep_size,
            moe_ep_rank=rank,
            tp_size=ep_size,
            attn_dp_size=ep_size,
            attn_dp_rank=rank,
            moe_tp_size=1,
            moe_dp_size=1,
            launch_world_rank=1,
        ),
    )


@pytest.mark.parametrize("kernel_type", ["IntraNode", "AsyncLL"])
@pytest.mark.parametrize(
    "sender_rows,rank,expected",
    [
        ([1] * 8, 0, 32),
        ([56] * 8, 3, 448),
        ([0, 1, 7, 33, 56, 128, 257, 448], 3, 960),
        ([0, 1, 7, 56], 0, 64),
        ([0, 56], 0, 64),
    ],
)
def test_epv1_recv_bound_uses_sender_sum_not_topk(
    monkeypatch, kernel_type, sender_rows, rank, expected
):
    _patch_epv1_recv_bound(monkeypatch, sender_rows=sender_rows, rank=rank)
    rows, reason = aiter_runner._mori_epv1_recv_bound(
        recv_rows=8 * 4096, local_rows=sender_rows[rank], kernel_type=kernel_type
    )
    assert (rows, reason) == (expected, "trimmed_dedup")


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"kernel_type": "InterNodeV1"}, "layout_unverified"),
        ({"kernel_type": "InterNodeV1LL"}, "layout_unverified"),
        ({"kernel_type": None}, "layout_unverified"),
        ({"local_rows": 8}, "metadata_mismatch"),
        ({"sender_rows": None}, "metadata_missing"),
        ({"sender_rows": [7] * 7}, "metadata_invalid"),
        ({"tbo": True}, "tbo_metadata_missing"),
        ({"recv_rows": 64}, "no_saving"),
        ({"recv_rows": 40}, "capacity_unproven"),
        ({"enabled": False}, "disabled"),
    ],
)
def test_epv1_recv_bound_keeps_full_view_when_unproved(monkeypatch, overrides, reason):
    sender_rows = overrides.get("sender_rows", [7] * 8)
    _patch_epv1_recv_bound(
        monkeypatch,
        sender_rows=sender_rows,
        tbo=overrides.get("tbo", False),
        ep_size=8,
    )
    if not overrides.get("enabled", True):
        monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "0")
    recv_rows = overrides.get("recv_rows", 32768)
    rows, got_reason = aiter_runner._mori_epv1_recv_bound(
        recv_rows=recv_rows,
        local_rows=overrides.get("local_rows", 7),
        kernel_type=overrides.get("kernel_type", "IntraNode"),
    )
    assert (rows, got_reason) == (recv_rows, reason)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
