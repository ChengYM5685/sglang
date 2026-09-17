"""DP gathers must agree on per-rank row counts after idle materialization."""

import unittest
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.layers import dp_attention, logits_processor
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDPGatherCountSource(unittest.TestCase):
    @contextmanager
    def _rank(self, rank, forward_counts):
        # Run the real GroupCoordinator.all_gatherv, including its equal-size
        # fast path. Only the communicator at the device boundary is mocked.
        comm = Mock(disabled=False)
        comm.change_state.return_value = nullcontext()
        group = SimpleNamespace(world_size=8, rank_in_group=rank, pynccl_comm=comm)
        group.all_gatherv = MethodType(GroupCoordinator.all_gatherv, group)
        flags = SimpleNamespace(
            dp=SimpleNamespace(
                capturing_prefill_graph=False,
                use_world_group_for_gather=False,
                buffer_hidden_size=2,
                buffer_dtype=torch.float32,
                buffer_device=torch.device("cpu"),
            )
        )
        with (
            patch.multiple(
                dp_attention,
                _USE_DP_GATHERV=True,
                get_flags=lambda: flags,
                get_attention_dp_size=lambda: 8,
                get_attention_dp_rank=lambda: rank,
                get_attn_tensor_model_parallel_world_size=lambda: 1,
                get_attn_tensor_model_parallel_rank=lambda: 0,
                get_tensor_model_parallel_world_size=lambda: 8,
                get_tp_group=lambda: group,
            ),
            patch.multiple(
                dp_attention._DpGatheredBufferWrapper,
                _global_num_tokens=forward_counts,
                _dp_max_padding=False,
            ),
            patch.object(
                logits_processor,
                "get_parallel",
                return_value=SimpleNamespace(attn_dp_rank=rank),
            ),
        ):
            yield comm

    def test_logits_counts_stay_consistent_across_active_and_idle_ranks(self):
        # A logits total equal to DP size used to accept the idle rank's
        # synthetic [1]*8 forward counts. Totals on either side must also work.
        for counts in (
            [1, 1, 0, 1, 1, 1, 1, 1],
            [2, 1, 0, 1, 1, 1, 1, 1],
            [3, 1, 0, 1, 1, 1, 1, 1],
        ):
            for rank in range(8):
                with self.subTest(counts=counts, rank=rank):
                    forward_counts = (
                        [1] * 8
                        if rank == 2
                        else [8192, 8192, 0, 8192, 8192, 8192, 8192, 8192]
                    )
                    local = torch.arange(
                        max(1, counts[rank]) * 2, dtype=torch.float32
                    ).reshape(-1, 2)
                    metadata = LogitsMetadata(
                        forward_mode=ForwardMode.EXTEND,
                        global_num_tokens_for_logprob_cpu=counts,
                        global_num_tokens_for_logprob_gpu=torch.tensor(counts),
                        dp_padding_mode=dp_attention.DpPaddingMode.SUM_LEN,
                    )
                    processor = SimpleNamespace(
                        do_tensor_parallel_all_gather_dp_attn=True
                    )
                    with self._rank(rank, forward_counts) as comm:
                        gathered, original = (
                            LogitsProcessor._gather_dp_attn_hidden_states(
                                processor, local, metadata
                            )
                        )

                    self.assertIs(original, local)
                    self.assertEqual(gathered.shape, (sum(counts), 2))
                    self.assertEqual(comm.all_gather.call_count, 1)
                    call = comm.all_gather.call_args
                    self.assertEqual(call.kwargs["sizes"], counts)
                    # Idle still participates, but its dummy row must not be
                    # sent as real LM-head input.
                    torch.testing.assert_close(call.args[1], local[: counts[rank]])

    def test_forward_gathers_keep_buffer_padding_and_equal_size_fast_path(self):
        metadata = SimpleNamespace(
            global_num_tokens_cpu=[1] * 8,
            global_num_tokens_for_logprob_cpu=[1] * 8,
            dp_padding_mode=dp_attention.DpPaddingMode.SUM_LEN,
        )
        local = torch.tensor([[4.0, 5.0]])
        for gather in (
            dp_attention.dp_gather_partial,
            dp_attention.dp_gather_replicate,
        ):
            with self.subTest(gather=gather.__name__):
                with self._rank(0, [2] * 8) as comm:
                    gather(torch.empty(16, 2), local, metadata)

                self.assertEqual(comm.all_gather.call_count, 1)
                call = comm.all_gather.call_args
                self.assertIsNone(call.kwargs["sizes"])
                torch.testing.assert_close(
                    call.args[1], torch.tensor([[4.0, 5.0], [0.0, 0.0]])
                )

    def test_missing_logits_cpu_counts_do_not_use_synthetic_forward_counts(self):
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            global_dp_buffer_len=8,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=torch.tensor([2, 1, 0, 1, 1, 1, 1, 1]),
            dp_padding_mode=dp_attention.DpPaddingMode.SUM_LEN,
        )
        processor = SimpleNamespace(do_tensor_parallel_all_gather_dp_attn=True)
        with (
            self._rank(2, [1] * 8) as comm,
            patch.object(dp_attention, "memcpy_func", dp_attention.memcpy_cpu),
            patch.object(
                dp_attention,
                "tensor_model_parallel_all_reduce",
                side_effect=lambda tensor: tensor,
            ) as all_reduce,
        ):
            gathered, _ = LogitsProcessor._gather_dp_attn_hidden_states(
                processor, torch.tensor([[9.0, 9.0]]), metadata
            )

        comm.all_gather.assert_not_called()
        self.assertEqual(all_reduce.call_count, 1)
        # This rank has zero real logits rows, so its all-reduce contribution
        # must be all zeros despite the materialized dummy input.
        torch.testing.assert_close(gathered, torch.zeros(8, 2))

    def test_invalid_logits_counts_fail_before_communication(self):
        for counts in (
            [1] * 7,  # missing rank
            [2, 1, 0, 1, 1, 1, 1, 2],  # wrong total
            [3, 1, -1, 1, 1, 1, 1, 1],  # negative count, matching total
        ):
            with self.subTest(counts=counts):
                metadata = SimpleNamespace(
                    global_num_tokens_for_logprob_cpu=counts,
                    dp_padding_mode=dp_attention.DpPaddingMode.SUM_LEN,
                )
                with self._rank(2, [1] * 8) as comm:
                    with self.assertRaisesRegex(ValueError, "DP gather logits counts"):
                        dp_attention.dp_gather_replicate(
                            torch.empty(8, 2),
                            torch.ones(1, 2),
                            metadata,
                            count_source="logits",
                        )
                comm.all_gather.assert_not_called()


if __name__ == "__main__":
    unittest.main()
