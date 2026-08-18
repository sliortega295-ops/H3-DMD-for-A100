from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from h3_a100.activation_offload import (
    SelectiveSavedTensorOffload,
    maybe_saved_tensor_offload,
)


class _Checkpointed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def body(x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x @ self.weight.t())

        return checkpoint(body, value, use_reentrant=False)


def test_disabled_scope_is_a_noop() -> None:
    module = _Checkpointed()
    value = torch.randn(2, 8, requires_grad=True)
    with maybe_saved_tensor_offload(
        module,
        enabled=False,
        logical_component="test",
        min_offload_bytes=128,
        pin_memory=False,
    ) as scope:
        assert scope is None
        module(value).sum().backward()


def test_scope_preserves_checkpointed_gradient_and_replay_contract() -> None:
    torch.manual_seed(7)
    reference = _Checkpointed()
    candidate = _Checkpointed()
    candidate.load_state_dict(reference.state_dict())
    ref_value = torch.randn(2, 8, requires_grad=True)
    test_value = ref_value.detach().clone().requires_grad_(True)

    ref_output = reference(ref_value)
    ref_output.sum().backward()

    scope = SelectiveSavedTensorOffload(
        candidate,
        logical_component="test",
        min_offload_bytes=0,
        pin_memory=False,
    )
    with scope.context():
        output = candidate(test_value)
        scope.begin_backward()
        output.sum().backward()

    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(test_value.grad, ref_value.grad)
    torch.testing.assert_close(candidate.weight.grad, reference.weight.grad)
    assert scope.stats["pack_count"] > 0
    assert scope.stats["replay_pack_count"] >= 0

