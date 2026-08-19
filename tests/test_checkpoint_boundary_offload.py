from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from h3_a100.checkpoint_boundary_offload import install_checkpoint_boundary_cpu_offload


class _ToyBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        return torch.tanh(self.proj(hidden)) + hidden


class _ToyTransformer(torch.nn.Module):
    def __init__(self, blocks: int = 3, width: int = 8):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_ToyBlock(width) for _ in range(blocks)])
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = lambda fn, *args, **kwargs: checkpoint(
            fn, *args, use_reentrant=False, **kwargs
        )

    def forward(self, hidden):
        for block in self.transformer_blocks:
            hidden = self._gradient_checkpointing_func(block, hidden)
        return hidden


def test_boundary_offload_preserves_output_and_gradients(tmp_path):
    torch.manual_seed(7)
    reference = _ToyTransformer()
    candidate = _ToyTransformer()
    candidate.load_state_dict(reference.state_dict())

    x_ref = torch.randn(2, 5, 8, requires_grad=True)
    x_candidate = x_ref.detach().clone().requires_grad_(True)

    y_ref = reference(x_ref)
    y_ref.square().mean().backward()

    role = {"value": "student"}
    registration = install_checkpoint_boundary_cpu_offload(
        candidate,
        role_getter=lambda: role["value"],
        event_path=tmp_path / "events.jsonl",
        pin_memory=False,
        expected_block_count=3,
        require_cuda=False,
    )
    y_candidate = candidate(x_candidate)
    y_candidate.square().mean().backward()

    torch.testing.assert_close(y_candidate, y_ref)
    torch.testing.assert_close(x_candidate.grad, x_ref.grad)
    for left, right in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(right.grad, left.grad)

    stats = registration.stats
    assert stats["grad_transformer_forward_count"] == 1
    assert stats["student_grad_forward_count"] == 1
    assert stats["grad_checkpoint_call_count"] == 3
    assert stats["cpu_copy_count"] == 3
    assert stats["offloaded_logical_bytes"] > 0
    assert (tmp_path / "events.jsonl").is_file()


def test_boundary_offload_tracks_shared_backbone_roles():
    torch.manual_seed(11)
    model = _ToyTransformer(blocks=2)
    role = {"value": "student"}
    registration = install_checkpoint_boundary_cpu_offload(
        model,
        role_getter=lambda: role["value"],
        pin_memory=False,
        expected_block_count=2,
        require_cuda=False,
    )

    x = torch.randn(1, 3, 8, requires_grad=True)
    model(x).sum().backward()
    role["value"] = "fake"
    x2 = torch.randn(1, 3, 8, requires_grad=True)
    model(x2).sum().backward()

    stats = registration.stats
    assert stats["student_grad_forward_count"] == 1
    assert stats["fake_grad_forward_count"] == 1
    assert stats["other_grad_forward_count"] == 0
    assert stats["grad_checkpoint_call_count"] == 4
    assert stats["cpu_copy_count"] == 4


def test_boundary_offload_does_not_touch_no_grad_forward():
    model = _ToyTransformer(blocks=2)
    registration = install_checkpoint_boundary_cpu_offload(
        model,
        role_getter=lambda: "teacher",
        pin_memory=False,
        expected_block_count=2,
        require_cuda=False,
    )
    with torch.no_grad():
        output = model(torch.randn(1, 3, 8))
    assert torch.isfinite(output).all()
    assert registration.stats["grad_transformer_forward_count"] == 0
    assert registration.stats["cpu_copy_count"] == 0


def test_boundary_offload_fails_closed_on_non_native_layout():
    model = _ToyTransformer(blocks=2)
    try:
        install_checkpoint_boundary_cpu_offload(
            model,
            role_getter=lambda: "student",
            pin_memory=False,
            expected_block_count=3,
            require_cuda=False,
        )
    except RuntimeError as exc:
        assert "native per-block layout" in str(exc)
    else:
        raise AssertionError("expected native checkpoint layout mismatch to fail")
