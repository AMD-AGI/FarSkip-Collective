
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Union, Tuple

import torch
import torch.distributed

from .parallel_state import get_tp_group

class CustomNCCLWork:
    """Work handle for async NCCL work following the torch.distributed API:
    https://pytorch.org/docs/stable/distributed.html#torch.distributed.Work"""

    def __init__(self, stream: torch.cuda.Stream, tensor: torch.Tensor):
        self.stream = stream
        self.tensor = tensor
        self._event = torch.cuda.Event()
        self._event.record(stream)

    def wait(self) -> torch.Tensor:
        """
        Uses stream.wait_event() to create a GPU-side stream dependency
        """
        torch.cuda.current_stream().wait_event(self._event)
        return self.tensor

    def is_completed(self) -> bool:
        return self._event.query()


def async_tensor_model_parallel_all_reduce(
    input_: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[CustomNCCLWork]]:
    """Async all-reduce with same kernel selection logic as synchronous version.
    mirrors the kernel selection priority from GroupCoordinator.all_reduce()

    Key differences from synchronous version:
    - Uses dedicated persistent streams (created once per communicator)
    - Quick/Custom AllReduce are out-of-place (return new tensor)
    - PyNccl is in-place (modifies input_ directly)
    """
    tp_group = get_tp_group()

    if tp_group.world_size == 1:
        return input_, None

    current_stream = torch.cuda.current_stream()

    qr_comm = tp_group.qr_comm
    if (
        qr_comm is not None
        and not qr_comm.disabled
        and qr_comm.should_quick_allreduce(input_)
    ):
        if not hasattr(qr_comm, 'stream'): # in AITER case
            qr_comm.stream = torch.cuda.Stream()
        qr_stream = qr_comm.stream
        qr_stream.wait_stream(current_stream)

        with torch.cuda.stream(qr_stream):
            output = qr_comm.quick_all_reduce(input_)

        return output, CustomNCCLWork(qr_stream, output)

    ca_comm = tp_group.ca_comm
    if (
        ca_comm is not None
        and not ca_comm.disabled
        and ca_comm.should_custom_ar(input_)
    ):
        if not hasattr(ca_comm, 'stream'): # in AITER case
            ca_comm.stream = torch.cuda.Stream()
        ca_stream = ca_comm.stream
        ca_stream.wait_stream(current_stream)

        with torch.cuda.stream(ca_stream):
            output = ca_comm.custom_all_reduce(input_)

        return output, CustomNCCLWork(ca_stream, output)

    pymscclpp_comm = tp_group.pymscclpp_comm
    if (
        pymscclpp_comm is not None
        and not pymscclpp_comm.disabled
        and pymscclpp_comm.should_mscclpp_allreduce(input_)
    ):
        pynccl_stream = tp_group.pynccl_comm.stream
        pynccl_stream.wait_stream(current_stream)

        with pymscclpp_comm.change_state(enable=True):
            with torch.cuda.stream(pynccl_stream):
                output = pymscclpp_comm.all_reduce(input_)

        return output, CustomNCCLWork(pynccl_stream, output)

    pynccl_stream = tp_group.pynccl_comm.stream
    pynccl_stream.wait_stream(current_stream)

    with tp_group.pynccl_comm.change_state(enable=True, stream=pynccl_stream):
        tp_group.pynccl_comm.all_reduce(input_)

    return input_, CustomNCCLWork(pynccl_stream, input_)

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
