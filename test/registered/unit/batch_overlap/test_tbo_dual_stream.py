"""SGLANG_TBO_DUAL_STREAM: TBO micro-batch B runs on a side stream, so work
queued behind a stalled micro-batch A (a MoE receive spinning on a slow rank)
does not hold B up, and the main stream gets B's results back. Needs a GPU.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap import operations as ops
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _forward_batch():
    return SimpleNamespace(
        global_dp_buffer_len=None,
        tbo_padded_len=None,
        global_num_tokens_cpu=None,
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
        tbo_child_global_num_tokens_cpu=None,
    )


def _run(stall_a: bool):
    """Two stages per micro-batch: x -> x*2 -> x+1. A optionally spins first."""
    seen = {}

    def stage1(state, x, forward_batch, name):
        seen[name] = torch.cuda.current_stream()
        if stall_a and name == "a":
            torch.cuda._sleep(2_000_000_000)  # ~1 s busy-wait on A's stream
        return dict(x=x * 2, forward_batch=forward_batch, name=name)

    def stage2(state, x, forward_batch, name):
        done = torch.cuda.Event()
        out = x + 1
        done.record()
        seen[name + "_done"] = done
        return dict(x=out)

    operations = [stage1, ops.YieldOperation(), stage2]
    inputs = [
        dict(x=torch.full((1024,), v, device="cuda"), forward_batch=_forward_batch(), name=n)
        for n, v in (("a", 1.0), ("b", 10.0))
    ]
    with patch.object(ops, "_resolve_tbo_child_contexts", return_value=(None, None)):
        out_a, out_b = ops.execute_overlapped_operations(
            inputs, [operations, operations], delta_stages=[0, 0]
        )
    return seen, out_a["x"], out_b["x"]


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestTboDualStream(CustomTestCase):
    def test_single_stream_by_default(self):
        with patch.dict(os.environ, {"SGLANG_TBO_DUAL_STREAM": "0"}):
            seen, a, b = _run(stall_a=False)
        self.assertEqual(seen["a"], seen["b"])
        torch.testing.assert_close(a, torch.full_like(a, 3.0))
        torch.testing.assert_close(b, torch.full_like(b, 21.0))

    def test_b_on_side_stream_and_results_reach_main(self):
        with patch.dict(os.environ, {"SGLANG_TBO_DUAL_STREAM": "1"}):
            seen, a, b = _run(stall_a=False)
        self.assertEqual(seen["a"], torch.cuda.current_stream())
        self.assertNotEqual(seen["b"], seen["a"])
        # Read on the main stream without an explicit sync: the main stream has
        # waited for the side stream.
        torch.testing.assert_close(b + 0, torch.full_like(b, 21.0))
        torch.testing.assert_close(a, torch.full_like(a, 3.0))

    def test_stalled_a_does_not_hold_up_b(self):
        with patch.dict(os.environ, {"SGLANG_TBO_DUAL_STREAM": "1"}):
            seen, a, b = _run(stall_a=True)
        seen["b_done"].synchronize()
        self.assertFalse(seen["a_done"].query(), "B finished only after A's stall")
        torch.cuda.synchronize()
        torch.testing.assert_close(b, torch.full_like(b, 21.0))

    def test_stalled_a_holds_up_b_on_one_stream(self):
        with patch.dict(os.environ, {"SGLANG_TBO_DUAL_STREAM": "0"}):
            seen, a, b = _run(stall_a=True)
        seen["b_done"].synchronize()
        self.assertTrue(seen["a_done"].query())


if __name__ == "__main__":
    unittest.main()
