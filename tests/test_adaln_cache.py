from __future__ import annotations

import torch
from torch import nn

from h3_a100.adaln_cache import (
    AdaLNProjectionBank,
    adaln_bank,
    install_adaln_cache,
)


class CountingProjection(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(offset))
        self.calls = 0

    def forward(self, temb: torch.Tensor):
        self.calls += 1
        value = temb + self.weight
        return tuple(value + index for index in range(6))


class DummyBlock(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.adaln_proj = CountingProjection(offset)


class DummyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([DummyBlock(1.0), DummyBlock(2.0)])
        self.time_embedder = nn.Linear(2, 2, bias=False)


def _installed():
    transformer = DummyTransformer()
    transformer.requires_grad_(False)
    controller = install_adaln_cache(transformer, max_dynamic_keys=1)
    bank = adaln_bank(transformer)
    assert isinstance(bank, AdaLNProjectionBank)
    return transformer, bank, controller


def test_persistent_key_computes_each_block_once_and_is_exact():
    transformer, bank, controller = _installed()
    temb = torch.tensor([[1.0, 2.0]])

    with controller.scope(("rollout", 0), persistent=True):
        first = [block.adaln_proj(temb) for block in transformer.transformer_blocks]
    with controller.scope(("rollout", 0), persistent=True):
        second = [block.adaln_proj(temb) for block in transformer.transformer_blocks]

    assert [projection.calls for projection in bank.projections] == [1, 1]
    for first_block, second_block in zip(first, second, strict=True):
        for first_tensor, second_tensor in zip(first_block, second_block, strict=True):
            torch.testing.assert_close(first_tensor, second_tensor, rtol=0, atol=0)

    stats = controller.stats()
    assert stats.misses == 2
    assert stats.hits == 2
    assert stats.persistent_keys == 1
    assert stats.bytes > 0


def test_dynamic_key_reuses_then_recomputes_after_drop():
    transformer, bank, controller = _installed()
    temb = torch.tensor([[3.0, 4.0]])
    key = ("score", 1)

    with controller.scope(key):
        transformer.transformer_blocks[0].adaln_proj(temb)
        transformer.transformer_blocks[0].adaln_proj(temb)
    assert bank.projections[0].calls == 1

    controller.drop(key)
    with controller.scope(key):
        transformer.transformer_blocks[0].adaln_proj(temb)
    assert bank.projections[0].calls == 2


def test_dynamic_lru_never_evicts_persistent_rollout():
    transformer, _bank, controller = _installed()
    temb = torch.tensor([[1.0, 1.0]])

    with controller.scope("persistent", persistent=True):
        transformer.transformer_blocks[0].adaln_proj(temb)
    with controller.scope("dynamic-1"):
        transformer.transformer_blocks[0].adaln_proj(temb)
    with controller.scope("dynamic-2"):
        transformer.transformer_blocks[0].adaln_proj(temb)

    stats = controller.stats()
    assert stats.persistent_keys == 1
    assert stats.dynamic_keys == 1
    assert stats.evictions == 1
    assert controller.has_complete_key("persistent", 1)


def test_install_rejects_trainable_adaln_weights():
    transformer = DummyTransformer()
    try:
        install_adaln_cache(transformer)
    except RuntimeError as error:
        assert "must be frozen" in str(error)
    else:
        raise AssertionError("trainable AdaLN projections must not be cached")


def test_install_is_idempotent():
    transformer, _bank, first = _installed()
    second = install_adaln_cache(transformer)
    assert first is second
