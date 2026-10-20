"""SGLANG_TBO_MIN_EXTEND_TOKENS keeps short prefill batches out of TBO.

Halving a short prefill doubles its kernel launches without overlapping
anything, so a rank whose extend batch is below the threshold votes against
TBO, and the all-gathered minimum turns it off for the whole step.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _extend_batch(num_tokens):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        spec_info=None,
        extend_num_tokens=num_tokens,
        extend_lens=[num_tokens // 2, num_tokens - num_tokens // 2],
        is_extend_in_batch=True,
    )


class TestTboMinExtendTokens(CustomTestCase):
    def _local_vote(self, batch):
        backend = SimpleNamespace(
            is_none=lambda: False, low_latency_supports_extend=lambda: True
        )
        with (
            patch.object(two_batch_overlap, "is_tbo_enabled", return_value=True),
            patch.object(
                two_batch_overlap, "get_deepep_mode", return_value=DeepEPMode.NORMAL
            ),
            patch.object(
                two_batch_overlap, "get_moe_a2a_backend", return_value=backend
            ),
            patch.object(two_batch_overlap, "compute_split_seq_index", return_value=1),
        ):
            preparer = TboDPAttentionPreparer()
            can_run_tbo, _ = preparer.prepare_all_gather(batch)
        return preparer, can_run_tbo

    def test_threshold(self):
        for threshold, num_tokens, expected in (
            (0, 16, True),
            (4096, 4095, False),
            (4096, 4096, True),
        ):
            with self.subTest(threshold=threshold, num_tokens=num_tokens):
                with envs.SGLANG_TBO_MIN_EXTEND_TOKENS.override(threshold):
                    _, can_run_tbo = self._local_vote(_extend_batch(num_tokens))
                self.assertEqual(can_run_tbo, expected)

    def test_idle_rank_does_not_block(self):
        with envs.SGLANG_TBO_MIN_EXTEND_TOKENS.override(4096):
            _, can_run_tbo = self._local_vote(None)
        self.assertTrue(can_run_tbo)

    def test_one_short_rank_unsplits_the_step(self):
        with envs.SGLANG_TBO_MIN_EXTEND_TOKENS.override(4096):
            preparer, _ = self._local_vote(_extend_batch(1024))
        mode = ForwardMode.EXTEND.value
        split, _ = preparer.compute_output(torch.tensor([[1, mode], [0, mode]]))
        self.assertIsNone(split)


if __name__ == "__main__":
    unittest.main()
