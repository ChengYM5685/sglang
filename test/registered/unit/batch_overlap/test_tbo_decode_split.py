"""SGLANG_ENABLE_TBO_DECODE: DeepSeek-V4 two-batch overlap on decode and DSpark verify.

With the switch off nothing changes: DFlash verify inputs stay unsplit and the
model gate keeps TBO to real prefill. With it on, the linear DFlash verify input is
split by request (eager, graph capture and graph replay alike), one-request graphs
stay whole on every rank, and each micro-batch's MoE receive bound is in tokens.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap import operations_strategy, two_batch_overlap
from sglang.srt.batch_overlap.operations import YieldOperation
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboCudaGraphRunnerPlugin,
    TboForwardBatchPreparer,
    _verify_input_blocks_split,
    split_spec_info,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_device
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

W = 7  # DSpark block 6 + the bonus token: draft_token_num rows per request.


def _verify_input(bs: int, **kwargs) -> DFlashVerifyInput:
    return DFlashVerifyInput(
        draft_token=torch.arange(bs * W),
        positions=torch.arange(bs * W) + 100,
        draft_token_num=W,
        live_seq_lens_cpu=torch.arange(bs) + 10,
        **kwargs,
    )


# A backend that verifies causally and never reads the verify mask, like DSV4 HIP.
_DSV4_LIKE = SimpleNamespace(dflash_verify_ignores_custom_mask=True)


def _decode_tbo(on: bool):
    return envs.SGLANG_ENABLE_TBO_DECODE.override(on)


def _tbo_on():
    return patch.object(two_batch_overlap, "is_tbo_enabled", return_value=True)


class TestDFlashVerifySplit(CustomTestCase):
    def test_dflash_verify_blocks_split_unless_switched_on(self):
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
            batch = SimpleNamespace(forward_mode=mode, spec_info=_verify_input(4))
            with self.subTest(mode=mode), _tbo_on():
                with _decode_tbo(False):
                    self.assertTrue(_verify_input_blocks_split(batch))
                with _decode_tbo(True):
                    self.assertFalse(_verify_input_blocks_split(batch))

    def test_masked_verify_splits_only_for_a_backend_that_ignores_the_mask(self):
        """Graph capture and idle replays hand DSV4 the runner's mask buffer; the
        mask is not cut per request, so only a backend that never reads it splits."""
        mask = torch.ones(4, dtype=torch.bool)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=_verify_input(4, custom_mask=mask),
        )
        for backend, blocks in (
            (SimpleNamespace(), True),
            (_DSV4_LIKE, False),
            (Mock(spec=TboAttnBackend, primary=_DSV4_LIKE), False),
        ):
            with (
                self.subTest(backend=backend),
                _tbo_on(),
                _decode_tbo(True),
                patch.object(
                    two_batch_overlap, "get_attn_backend", return_value=backend
                ),
            ):
                self.assertEqual(_verify_input_blocks_split(batch), blocks)

    def test_ragged_verify_still_blocks(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=_verify_input(4, ragged_verify_layout=object()),
        )
        with _tbo_on(), _decode_tbo(True):
            self.assertTrue(_verify_input_blocks_split(batch))

    def test_split_by_request(self):
        spec_info = _verify_input(5)
        with _tbo_on(), _decode_tbo(True):
            child = split_spec_info(
                spec_info=spec_info,
                start_seq_index=2,
                end_seq_index=5,
                start_token_index=2 * W,
                end_token_index=5 * W,
            )
        self.assertIsInstance(child, DFlashVerifyInput)
        self.assertEqual(child.draft_token_num, W)
        self.assertEqual(child.num_tokens_per_req, W)
        self.assertEqual(child.draft_token.tolist(), list(range(2 * W, 5 * W)))
        self.assertEqual(
            child.positions.tolist(), list(range(2 * W + 100, 5 * W + 100))
        )
        self.assertEqual(child.live_seq_lens_cpu.tolist(), [12, 13, 14])
        self.assertIsNone(child.iteration_cols)
        # The parent is not modified.
        self.assertEqual(spec_info.draft_token.shape[0], 5 * W)

    def test_split_capture_input_with_unset_tokens(self):
        # Graph capture builds the verify input with draft_token / positions None.
        spec_info = DFlashVerifyInput(
            draft_token=None, positions=None, draft_token_num=W
        )
        with _tbo_on(), _decode_tbo(True):
            child = split_spec_info(
                spec_info=spec_info,
                start_seq_index=0,
                end_seq_index=2,
                start_token_index=0,
                end_token_index=2 * W,
            )
        self.assertIsNone(child.draft_token)
        self.assertIsNone(child.positions)
        self.assertIsNone(child.live_seq_lens_cpu)


def _capture_verify_batch(bs: int, dp_size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=DFlashVerifyInput(
            draft_token=None, positions=None, draft_token_num=W
        ),
        batch_size=bs,
        extend_seq_lens_cpu=None,
        global_num_token_non_padded_cpu=None,
        input_ids=torch.zeros(bs * W, dtype=torch.long),
        global_num_tokens_cpu=[bs * W] * dp_size,
        global_tbo_split_num_tokens=None,
        tbo_split_seq_index=None,
    )


class TestDecodeTboGraphCapture(CustomTestCase):
    def _capture(self, batch, *, decode_tbo: bool) -> Mock:
        with get_context().override_server_args():
            with get_device().override(device="meta"):
                plugin = TboCudaGraphRunnerPlugin()
        with (
            _tbo_on(),
            _decode_tbo(decode_tbo),
            patch.object(TboForwardBatchPreparer, "prepare_raw") as prepare_raw,
            patch.object(
                TboForwardBatchPreparer,
                "compute_tbo_children_num_token_non_padded",
                return_value=torch.zeros(2, dtype=torch.int32, device="meta"),
            ),
        ):
            plugin.capture_one_batch_size(batch, num_tokens=batch.batch_size * W)
        return prepare_raw

    def test_dflash_verify_graph_splits_with_exact_child_rows(self):
        batch = _capture_verify_batch(bs=9)
        prepare_raw = self._capture(batch, decode_tbo=True)
        prepare_raw.assert_called_once()
        self.assertEqual(batch.tbo_split_seq_index, 4)
        # Every rank's first micro-batch is 4 requests of W rows in this graph.
        self.assertEqual(batch.global_tbo_split_num_tokens, [4 * W] * 8)

    def test_one_request_graph_is_captured_whole(self):
        batch = _capture_verify_batch(bs=1)
        self._capture(batch, decode_tbo=True).assert_not_called()
        self.assertIsNone(batch.tbo_split_seq_index)

    def test_switch_off_keeps_dflash_verify_graph_whole(self):
        batch = _capture_verify_batch(bs=9)
        self._capture(batch, decode_tbo=False).assert_not_called()
        self.assertIsNone(batch.global_tbo_split_num_tokens)


class TestDecodeTboGraphReplay(CustomTestCase):
    def _replay_view(self, bs: int, raw_tokens: int) -> SimpleNamespace:
        return SimpleNamespace(
            batch_size=bs,
            forward_mode=ForwardMode.TARGET_VERIFY,
            actual_forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=_verify_input(bs),
            input_ids=torch.arange(bs * W),
            req_pool_indices=torch.arange(bs),
            seq_lens=torch.arange(bs) + 10,
            seq_lens_cpu=torch.arange(bs) + 10,
            # The live (unpadded) write locations of the replay.
            out_cache_loc=torch.arange(raw_tokens) + 1000,
            iteration_ids=None,
            iteration_cols=None,
            max_seq_len_override=None,
            encoder_lens=None,
        )

    def test_children_join_the_decode_graphs_under_the_switch(self):
        backend = SimpleNamespace(
            primary=SimpleNamespace(tbo_supports_cuda_graph=False)
        )
        with _tbo_on():
            with _decode_tbo(False):
                self.assertFalse(TboAttnBackend._children_use_cuda_graph(backend))
            with _decode_tbo(True):
                self.assertTrue(TboAttnBackend._children_use_cuda_graph(backend))

    def test_replay_splits_padded_verify_view_by_request(self):
        children = [Mock(), Mock()]
        backend = SimpleNamespace(children=children, primary=_DSV4_LIKE)
        # A 4-request graph replaying 3 live requests: the second child is padding
        # past the live write locations.
        view = self._replay_view(bs=4, raw_tokens=3 * W)
        with _tbo_on(), _decode_tbo(True):
            TboAttnBackend._dispatch_children_from_replay_view(backend, view)
        views = [c.init_forward_metadata_out_graph.call_args.kwargs for c in children]
        child_a, child_b = (v["forward_batch"] for v in views)
        self.assertEqual([child_a.batch_size, child_b.batch_size], [2, 2])
        self.assertEqual(child_a.spec_info.draft_token.shape[0], 2 * W)
        self.assertEqual(child_b.spec_info.live_seq_lens_cpu.tolist(), [12, 13])
        self.assertEqual(child_a.out_cache_loc.shape[0], 2 * W)
        self.assertEqual(child_b.out_cache_loc.shape[0], W)
        self.assertFalse(any(v["in_capture"] for v in views))

    def test_idle_replay_with_the_mask_buffer_splits(self):
        """An idle rank's replay carries the runner's mask buffer on its verify input
        and must split like the busy ranks, or the MoE collectives do not pair up."""
        children = [Mock(), Mock()]
        backend = SimpleNamespace(children=children, primary=_DSV4_LIKE)
        view = self._replay_view(bs=4, raw_tokens=0)
        view.actual_forward_mode = ForwardMode.IDLE
        mask = torch.ones(64, dtype=torch.bool)
        view.spec_info = DFlashVerifyInput(
            draft_token=torch.empty(0, dtype=torch.long),
            positions=torch.empty(0, dtype=torch.long),
            draft_token_num=W,
            custom_mask=mask,
        )
        with _tbo_on(), _decode_tbo(True):
            TboAttnBackend._dispatch_children_from_replay_view(backend, view)
        for child in children:
            child_view = child.init_forward_metadata_out_graph.call_args.kwargs[
                "forward_batch"
            ]
            self.assertIs(child_view.spec_info.custom_mask, mask)
            self.assertEqual(child_view.actual_forward_mode, ForwardMode.IDLE)

    def test_replay_of_whole_one_request_graph_skips_children(self):
        children = [Mock(), Mock()]
        backend = SimpleNamespace(children=children, primary=_DSV4_LIKE)
        with _tbo_on(), _decode_tbo(True):
            TboAttnBackend._dispatch_children_from_replay_view(
                backend, self._replay_view(bs=1, raw_tokens=W)
            )
        for child in children:
            child.init_forward_metadata_out_graph.assert_not_called()


class TestVerifySplitRowsInTokens(CustomTestCase):
    def test_gathered_split_rows_scale_like_the_token_counts(self):
        """The scheduler gathers a decode step's split in requests; a verify forward
        bounds each micro-batch's MoE receive in tokens, W per request."""
        empty = torch.zeros(0, dtype=torch.long)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=0,
            input_ids=empty,
            req_pool_indices=empty,
            seq_lens=empty,
            out_cache_loc=empty,
            seq_lens_sum=0,
            spec_info=_verify_input(0),
            global_tbo_split_num_tokens=[2, 0, 5],
        )
        schedule_batch = SimpleNamespace(
            dp_spec_prefill_coordination_applied=False,
            global_num_tokens=[4, 0, 9],
            global_num_tokens_for_logprob=[4, 0, 9],
            draft_global_num_tokens=None,
            can_run_decode_cuda_graph=True,
            can_run_dp_draft_cuda_graph=False,
        )
        forward_batch.init_mlp_sync_metadata(schedule_batch, "cpu")
        self.assertEqual(forward_batch.global_num_tokens_cpu, [4 * W, 0, 9 * W])
        self.assertEqual(forward_batch.global_tbo_split_num_tokens, [2 * W, 0, 5 * W])


def _layer() -> SimpleNamespace:
    return SimpleNamespace(
        op_mhc_prepare_attn=Mock(),
        self_attn=SimpleNamespace(op_attn=Mock()),
        op_mhc_post_attn_pre_mlp=Mock(),
        op_mhc_postprocess=Mock(),
        mlp=SimpleNamespace(
            **{
                name: Mock()
                for name in (
                    "op_gate",
                    "op_select_experts",
                    "op_dispatch_a",
                    "op_dispatch_b",
                    "op_experts",
                    "op_combine_a",
                    "op_shared_experts",
                    "op_combine_b",
                    "op_output",
                )
            }
        ),
    )


class TestDeepseekV4DecodeStrategy(CustomTestCase):
    def _operations(self, layer, mode, *, ep: bool):
        backend = Mock(is_none=Mock(return_value=not ep))
        with patch("sglang.srt.layers.moe.get_moe_a2a_backend", return_value=backend):
            return operations_strategy._compute_moe_deepseek_v4_layer_operations_strategy_tbo(
                layer, mode
            )

    def test_decode_and_verify_take_the_ep_prefill_schedule(self):
        layer = _layer()
        prefill = self._operations(layer, ForwardMode.EXTEND, ep=True)
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                strategy = self._operations(layer, mode, ep=True)
                self.assertEqual(strategy.tbo_delta_stages, 0)
                self.assertEqual(
                    [_op_key(op) for op in strategy.operations],
                    [_op_key(op) for op in prefill.operations],
                )

    def test_non_ep_decode_is_refused(self):
        with self.assertRaises(NotImplementedError):
            self._operations(_layer(), ForwardMode.DECODE, ep=False)


def _op_key(op):
    return "yield" if isinstance(op, YieldOperation) else id(op)


if __name__ == "__main__":
    unittest.main()
