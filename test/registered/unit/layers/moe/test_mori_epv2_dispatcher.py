"""CPU-only tests for how the MORI EPv2 dispatcher drives its transport.

The MORI op is a fake that logs its calls, so these pin which calls each mode
makes and when, not what the kernels compute.
"""

import inspect
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher import moriep
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    CombineDtype,
    DispatchDtype,
    MoriEPDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.runtime_context import get_flags, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HIDDEN = 8
_TOPK = 2


def _fake_mori_modules():
    """Just enough of MORI for the EPv2 impls' import probe."""
    mori = ModuleType("mori")
    cco = ModuleType("mori.cco")
    cco.Communicator = object
    ops = ModuleType("mori.ops")
    v2 = ModuleType("mori.ops.dispatch_combine_v2")
    v2.EpDispatchCombineOp = object
    return {
        "mori": mori,
        "mori.cco": cco,
        "mori.ops": ops,
        "mori.ops.dispatch_combine_v2": v2,
    }


class _FakeOp:
    def __init__(self, log):
        self.log = log
        self.cfg = SimpleNamespace(
            effective_max_recv=64, is_internode=False, is_scatter=False
        )
        self.backend_name = "hip"
        self.dispatched = None
        self.combined = None

    def _dispatch_result(self, hidden, weights, indices):
        self.dispatched = (
            hidden.clone(),
            weights.clone(),
            None,
            indices.clone(),
            torch.tensor([hidden.shape[0]]),
            "routing",
        )
        return self.dispatched

    def dispatch(self, hidden, weights, scales, indices, **kwargs):
        self.log.append("dispatch")
        return self._dispatch_result(hidden, weights, indices)

    def dispatch_send(self, hidden, weights, scales, indices, return_routing):
        self.log.append("dispatch_send")
        return self._dispatch_result(hidden, weights, indices)

    def dispatch_recv(self):
        self.log.append("dispatch_recv")

    def combine(self, hidden, weights, indices, routing):
        self.log.append("combine")
        self.combined = hidden.clone()
        return self.combined, None

    def combine_send(self, hidden, weights, indices, routing):
        self.log.append("combine_send")
        self.combined = hidden.clone()
        return self.combined, None

    def combine_recv(self):
        self.log.append("combine_recv")


class _MoriEPv2TestCase(CustomTestCase):
    def setUp(self):
        reset_context()
        self.log = []
        self._patches = [
            patch.dict(sys.modules, _fake_mori_modules()),
            patch.dict(os.environ),
            patch.object(moriep, "get_is_extend_in_batch", return_value=False),
            patch.object(
                moriep, "_should_record_expert_distribution", return_value=False
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ]
        for item in self._patches:
            item.start()
        os.environ.pop("MORI_ENABLE_SDMA", None)
        envs.SGLANG_MORI_EP_V2.set(True)

    def tearDown(self):
        for item in reversed(self._patches):
            item.stop()
        reset_context()

    def _make_dispatcher(self, deepep_mode, async_finish=False):
        dispatcher = MoriEPDispatcher(
            group=None,
            router_topk=_TOPK,
            hidden_size=_HIDDEN,
            params_dtype=torch.bfloat16,
            deepep_mode=deepep_mode,
            async_finish=async_finish,
        )
        dispatcher._get_impl()._mori_op = _FakeOp(self.log)
        return dispatcher

    @staticmethod
    def _inputs(num_tokens=3):
        hidden = torch.randn(num_tokens, _HIDDEN, dtype=torch.bfloat16)
        ids = torch.randint(0, 4, (num_tokens, _TOPK), dtype=torch.int32)
        weights = torch.rand(num_tokens, _TOPK)
        return hidden, StandardTopKOutput(weights, ids, None)


class TestMoriEPv2Transport(_MoriEPv2TestCase):
    def test_low_latency_sends_in_a_and_receives_in_b(self):
        """TBO overlaps the sibling batch with the transfer only if the send half
        runs in *_a and the receive half in *_b; both halves in *_b still give the
        right result but lose the overlap."""
        dispatcher = self._make_dispatcher(DeepEPMode.LOW_LATENCY)
        op = dispatcher._get_impl().mori_op
        hidden, topk_output = self._inputs()

        dispatcher.dispatch_a(hidden, topk_output)
        self.log.append("sibling")
        dispatched = dispatcher.dispatch_b()
        self.assertEqual(self.log, ["dispatch_send", "sibling", "dispatch_recv"])
        # dispatch_b hands out the send half's buffers, which the receive fills.
        self.assertIs(dispatched.hidden_states, op.dispatched[0])
        self.assertIs(dispatched.topk_ids, op.dispatched[3])

        del self.log[:]
        dispatcher.combine_a(
            (dispatched.hidden_states, dispatched.topk_ids, dispatched.topk_weights)
        )
        self.log.append("sibling")
        combined = dispatcher.combine_b()
        self.assertEqual(self.log, ["combine_send", "sibling", "combine_recv"])
        self.assertIs(combined, op.combined)

    def test_sdma_normal_runs_both_halves_in_b_without_a_comm_stream(self):
        """SDMA normal mode reaches the copy engine through the same send/recv
        kernels, back to back. The send half returns at once, so a TBO comm stream
        would only add an event round trip."""
        os.environ["MORI_ENABLE_SDMA"] = "1"
        with get_flags().moe.override(tbo_enabled=True):
            dispatcher = self._make_dispatcher(DeepEPMode.NORMAL, async_finish=True)
        impl = dispatcher._get_impl()
        self.assertIsNone(impl._comm_stream)
        # MORI sizes the AsyncLL grids from the CU count; the TBO caps are tuned
        # for the intranode kernels.
        self.assertEqual(impl._launch_config, moriep._get_epv2_launch_config(False))
        hidden, topk_output = self._inputs()

        dispatcher.dispatch_a(hidden, topk_output)
        self.assertEqual(self.log, [])
        dispatched = dispatcher.dispatch_b()
        self.assertEqual(self.log, ["dispatch_send", "dispatch_recv"])

        del self.log[:]
        dispatcher.combine(
            (dispatched.hidden_states, dispatched.topk_ids, dispatched.topk_weights)
        )
        self.assertEqual(self.log, ["combine_send", "combine_recv"])

    def test_recv_bound_skips_async_ll_layouts(self):
        """mori_recv_bound is proven only for the intranode receive layout;
        trimming an AsyncLL receive view could drop received rows."""
        for deepep_mode, sdma, verified in (
            (DeepEPMode.NORMAL, False, True),
            (DeepEPMode.NORMAL, True, False),
            (DeepEPMode.LOW_LATENCY, False, False),
        ):
            with self.subTest(deepep_mode=deepep_mode, sdma=sdma):
                os.environ["MORI_ENABLE_SDMA"] = "1" if sdma else "0"
                impl = self._make_dispatcher(deepep_mode)._get_impl()
                self.assertEqual(impl._is_recv_layout_verified(), verified)


class TestMoriEPv2OpConfig(_MoriEPv2TestCase):
    def _build_op(self, deepep_mode, weight_dtype, sdma=False, combine="auto"):
        """Run set_quant_config and return init_mori_epv2_op's bound arguments."""
        os.environ["MORI_ENABLE_SDMA"] = "1" if sdma else "0"
        bound = {}
        signature = inspect.signature(moriep.init_mori_epv2_op)

        def fake_init(*args, **kwargs):
            bound.update(signature.bind(*args, **kwargs).arguments)
            return _FakeOp(self.log)

        dispatcher = MoriEPDispatcher(
            group=None,
            router_topk=_TOPK,
            hidden_size=_HIDDEN,
            params_dtype=torch.bfloat16,
            deepep_mode=deepep_mode,
        )
        impl = dispatcher._get_impl()
        # Stands in for the aiter quantizers a quantized dispatch requires.
        impl.fp4_quant_func = object()
        with (
            patch.object(moriep, "init_mori_epv2_op", fake_init),
            envs.SGLANG_MORI_COMBINE_DTYPE.override(combine),
            envs.SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS.override(1024),
        ):
            dispatcher.set_quant_config({"weight_dtype": weight_dtype})
        return bound

    def test_op_receives_mode_and_capacity_by_position(self):
        """_initialize_op passes init_mori_epv2_op's arguments positionally; a
        reordered signature would silently swap the mode, SDMA flag and receive
        capacity."""
        bound = self._build_op(DeepEPMode.LOW_LATENCY, torch.bfloat16, sdma=True)
        self.assertIs(bound["deepep_mode"], DeepEPMode.LOW_LATENCY)
        self.assertIs(bound["enable_sdma"], True)
        self.assertEqual(bound["max_total_recv_tokens"], 1024)
        self.assertEqual(
            bound["num_max_dispatch_tokens_per_rank"],
            envs.SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
        )

    def test_async_ll_restricts_transport_dtypes(self):
        """AsyncLL moves bf16 or fp8 tokens and combines unquantized; the
        intranode kernels keep every dtype."""
        fp4, fp8, bf16 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.bfloat16
        ll, normal = DeepEPMode.LOW_LATENCY, DeepEPMode.NORMAL
        for mode, weight_dtype, sdma, combine, expected in (
            # fp4 dispatch falls back to bf16 on both AsyncLL entries.
            (ll, fp4, False, "auto", ("bf16", "bf16")),
            (normal, fp4, True, "auto", ("bf16", "bf16")),
            # fp8 dispatch is in AsyncLL's range and stays.
            (ll, fp8, False, "auto", ("fp8", "bf16")),
            # A quantized combine falls back only on AsyncLL.
            (ll, bf16, False, "fp8", ("bf16", "bf16")),
            (normal, bf16, False, "fp8", ("bf16", "fp8")),
            (normal, fp4, False, "auto", ("fp4", "bf16")),
        ):
            with self.subTest(
                mode=mode, weight=weight_dtype, sdma=sdma, combine=combine
            ):
                bound = self._build_op(mode, weight_dtype, sdma=sdma, combine=combine)
                self.assertEqual(
                    (bound["dispatch_dtype"], bound["combine_dtype"]),
                    (DispatchDtype[expected[0]], CombineDtype[expected[1]]),
                )


class TestMoriEPv2TboPerCall(_MoriEPv2TestCase):
    """--enable-two-batch-overlap must only change the calls that belong to a TBO
    micro-batch; decode, verify and unsplit prefill forwards of the same server
    keep the ordinary transfer settings."""

    def _make_tbo_dispatcher(self):
        stream = object()
        with (
            get_flags().moe.override(tbo_enabled=True),
            patch.object(
                moriep.CommStreamPool, "get_stream_from_pool", return_value=stream
            ),
        ):
            dispatcher = self._make_dispatcher(DeepEPMode.NORMAL, async_finish=True)
        self.assertIs(dispatcher._get_impl()._comm_stream, stream)
        return dispatcher

    def test_maybe_tbo_dispatcher_marks_micro_batch_calls(self):
        from sglang.srt.batch_overlap.two_batch_overlap import (
            MaybeTboDeepEPDispatcher,
        )

        inners = [SimpleNamespace(dispatch_a=lambda **kw: kw) for _ in range(2)]
        wrapper = MaybeTboDeepEPDispatcher.__new__(MaybeTboDeepEPDispatcher)
        wrapper._inners = inners
        self.assertEqual(wrapper.dispatch_a(tbo_subbatch_index=1, x=1), {"x": 1})
        self.assertTrue(inners[1].tbo_subbatch_active)
        wrapper.dispatch_a(tbo_subbatch_index=0)
        self.assertTrue(inners[0].tbo_subbatch_active)
        wrapper.dispatch_a()
        self.assertFalse(inners[0].tbo_subbatch_active)

    def test_comm_stream_only_for_micro_batches(self):
        dispatcher = self._make_tbo_dispatcher()
        dispatcher.tbo_subbatch_active = True
        self.assertTrue(dispatcher._get_impl()._use_comm_stream())

        # An ordinary forward runs the transfer on the compute stream: no event,
        # no comm-stream hop.
        dispatcher.tbo_subbatch_active = False
        impl = dispatcher._get_impl()
        self.assertFalse(impl._use_comm_stream())
        self.assertIsNone(impl._capture_event_if_async())
        hidden, topk_output = self._inputs()
        dispatched = dispatcher.dispatch(hidden, topk_output)
        dispatcher.combine(
            (dispatched.hidden_states, dispatched.topk_ids, dispatched.topk_weights)
        )
        self.assertEqual(self.log, ["dispatch", "combine"])

    def test_recv_bound_only_skipped_for_micro_batches(self):
        """The DP token counts behind mori_recv_bound describe the whole batch, so
        only a micro-batch has to keep the full receive view."""
        envs.SGLANG_MORI_RECV_BOUND.set(True)
        dispatcher = self._make_tbo_dispatcher()
        with (
            patch.object(moriep, "mori_recv_bound", return_value=8),
            patch.object(moriep, "round_logical_recv_rows", return_value=16),
        ):
            for active, expected in ((False, 16), (True, 0)):
                with self.subTest(tbo_subbatch_active=active):
                    dispatcher.tbo_subbatch_active = active
                    impl = dispatcher._get_impl()
                    impl._num_tokens = 3
                    impl._dispatch_sender_rows = (3, 3)
                    self.assertEqual(impl._select_recv_cap(), expected)

    def test_grid_cap_switch(self):
        capped = moriep._get_epv2_launch_config(True)
        self.assertEqual(capped.dispatch_block_num, 32)
        self.assertEqual(capped.combine_block_num, 48)
        envs.SGLANG_MORI_EPV2_TBO_GRID_CAP.set(False)
        self.assertEqual(
            moriep._get_epv2_launch_config(True), moriep._get_epv2_launch_config(False)
        )


if __name__ == "__main__":
    unittest.main()
