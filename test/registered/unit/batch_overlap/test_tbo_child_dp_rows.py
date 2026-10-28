"""Per-rank rows that bound each TBO micro-batch's MORI dispatch.

Every rank must agree on each micro-batch's bound: a receiver caps its MoE
input by the sum, so one rank under-counting a sender's micro-batch drops rows.
An idle or empty-split rank starts BOTH children at token 0, so the child must
be identified by position, not by its start token. CPU-only.
"""

import unittest

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_rows = TboForwardBatchPreparer._child_dp_global_num_tokens


class TestTboChildDpRows(CustomTestCase):
    def test_children_partition_parent_rows(self):
        parent, split = [8192, 0, 6254, 3000], [4096, 0, 3127, 1400]
        first = _rows(parent, split, first_child=True)
        second = _rows(parent, split, first_child=False)
        self.assertEqual(first, [4096, 0, 3127, 1400])
        self.assertEqual(second, [4096, 0, 3127, 1600])
        self.assertEqual([a + b for a, b in zip(first, second)], parent)

    def test_split_is_clamped_to_parent_rows(self):
        self.assertEqual(_rows([10, 4], [12, -1], first_child=True), [10, 0])
        self.assertEqual(_rows([10, 4], [12, -1], first_child=False), [0, 4])

    def test_missing_or_mismatched_split_keeps_parent_rows(self):
        self.assertEqual(_rows([8, 8], None, first_child=False), [8, 8])
        self.assertEqual(_rows([8, 8], [4], first_child=True), [8, 8])
        self.assertIsNone(_rows(None, [4, 4], first_child=True))


if __name__ == "__main__":
    unittest.main()
