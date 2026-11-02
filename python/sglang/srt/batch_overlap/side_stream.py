"""Runs two-batch overlap's micro-batch B on its own stream, for MoE a2a backends
whose receive holds up the compute stream (see MoeA2ABackend)."""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.layers.moe.utils import get_moe_a2a_backend

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.operations import _StageExecutor

_side_streams: Dict[int, torch.cuda.Stream] = {}
_side_issuers: Dict[int, ThreadPoolExecutor] = {}


def tbo_side_stream() -> Optional[torch.cuda.Stream]:
    """The stream micro-batch B runs on, or None to keep both on the current one."""
    if (
        not get_moe_a2a_backend().tbo_micro_batch_b_on_side_stream()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    device = torch.cuda.current_device()
    if device not in _side_streams:
        _side_streams[device] = torch.cuda.Stream(device=device)
    return _side_streams[device]


class SideStreamExecutor:
    """Runs a _StageExecutor's stages on `stream`, issued from the device's side
    issuer thread; the caller waits for each stage to be issued, so the host still
    issues A's and B's stages in the same order."""

    def __init__(
        self, executor: _StageExecutor, stream: torch.cuda.Stream, inputs: dict
    ):
        self._executor = executor
        self._stream = stream
        self._main_stream = torch.cuda.current_stream()
        self._issuer = _side_issuer(stream.device.index)
        stream.wait_stream(self._main_stream)
        # B's inputs come from the main stream's pool: keep A from reusing their
        # blocks while B may still read them.
        _record_stream(inputs, stream)

    def next(self):
        inference_mode = torch.is_inference_mode_enabled()
        grad_enabled = torch.is_grad_enabled()

        def issue():
            with (
                torch.inference_mode(inference_mode),
                torch.set_grad_enabled(grad_enabled),
                torch.cuda.stream(self._stream),
            ):
                self._executor.next()

        # B reads the caller's context variables; its writes stay in the copy.
        self._issuer.submit(contextvars.copy_context().run, issue).result()

    def join(self):
        """Make the main stream wait for B; call after B's last stage."""
        self._main_stream.wait_stream(self._stream)
        # B's outputs come from the side stream's pool; the main stream uses them.
        _record_stream(self._executor.output, self._main_stream)

    def __getattr__(self, item):
        return getattr(self._executor, item)


def _record_stream(values, stream: torch.cuda.Stream) -> None:
    """record_stream the tensors in `values` (a dict, list or tuple, one level of
    nesting), so the caching allocator waits for `stream` before reusing them."""
    items = values.values() if isinstance(values, dict) else values
    for value in items:
        tensors = value if isinstance(value, (list, tuple)) else (value,)
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                tensor.record_stream(stream)


def _side_issuer(device: int) -> ThreadPoolExecutor:
    """The device's host thread that issues micro-batch B's stages.

    torch keeps one BLAS handle per host thread, and hipBLASLt Stream-K GEMMs
    running concurrently on two streams through one handle can leave a workgroup
    spinning in its fixup forever; issuing B from its own thread gives B its own
    handle. Drop the thread once hipBLASLt Stream-K is safe across streams on one
    handle.
    """
    issuer = _side_issuers.get(device)
    if issuer is None:
        issuer = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tbo-side-issuer",
            initializer=torch.cuda.set_device,
            initargs=(device,),
        )
        issuer.submit(_create_blas_handle).result()
        _side_issuers[device] = issuer
    return issuer


def _create_blas_handle() -> None:
    # Create this thread's BLAS handle on an idle device, so its setup cannot race
    # GEMMs in flight on other streams.
    torch.cuda.synchronize()
    x = torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16)
    torch.nn.functional.linear(x, x)
    torch.cuda.synchronize()
