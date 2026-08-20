from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from h3_a100.grid_adaln import (
    SCHEMA_VERSION,
    GridAdaLNController,
    install_grid_adaln_table,
)


def _bits(values: torch.Tensor):
    return [int(v) for v in values.to(torch.float32).contiguous().view(torch.int32).tolist()]


def _write_tiny_table(tmp_path):
    num_entries, num_blocks, hidden = 5, 50, 2
    shape = (num_entries, num_blocks, 6, 6, hidden)
    binary = tmp_path / "table.bin"
    numel = 1
    for value in shape:
        numel *= value
    table = torch.from_file(str(binary), shared=True, size=numel, dtype=torch.bfloat16).reshape(shape)
    for entry in range(num_entries):
        table[entry].fill_(float(entry + 1))
    del table
    pairs = [torch.tensor([0.1 + i * 0.01, 0.2 + i * 0.01], dtype=torch.float32) for i in range(num_entries)]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "binary_file": binary.name,
        "grid_size": 1,
        "num_rollout_entries": 4,
        "num_entries": num_entries,
        "num_blocks": num_blocks,
        "hidden_size": hidden,
        "rows_per_entry": 6,
        "modulation_chunks": 6,
        "dtype": "torch.bfloat16",
        "renoise_sigma_min": 0.02,
        "renoise_sigma_max": 0.98,
        "video_shift": 6.0,
        "audio_shift": 3.0,
        "num_inference_steps": 4,
        "timestep_bits": [_bits(pair) for pair in pairs],
    }
    path = tmp_path / "table.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, pairs


def test_grid_controller_loads_only_selected_entry(tmp_path):
    manifest, pairs = _write_tiny_table(tmp_path)
    controller = GridAdaLNController(manifest, max_dynamic_keys=1, pin_memory=False)
    controller.ensure_key(("score", 1), pairs[4], persistent=False)
    with controller.scope(("score", 1), persistent=False):
        chunks = controller.lookup(7)
    assert len(chunks) == 6
    assert chunks[0].shape == (6, 2)
    assert torch.all(chunks[0] == torch.tensor(5.0, dtype=torch.bfloat16))
    assert controller.stats().stores == 1
    assert controller.stats().hits == 1


def test_grid_controller_rejects_non_grid_timestep(tmp_path):
    manifest, _ = _write_tiny_table(tmp_path)
    controller = GridAdaLNController(manifest, pin_memory=False)
    with pytest.raises(RuntimeError, match="not represented"):
        controller.ensure_key(
            ("score", 9), torch.tensor([0.123456, 0.654321], dtype=torch.float32), persistent=False
        )


def test_grid_install_removes_adaln_parameters(tmp_path):
    manifest, pairs = _write_tiny_table(tmp_path)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.adaln_proj = nn.Linear(2, 3, bias=True)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([Block() for _ in range(50)])

    transformer = Transformer()
    before = sum(p.numel() for p in transformer.parameters())
    controller = install_grid_adaln_table(
        transformer, manifest, max_dynamic_keys=1, pin_memory=False
    )
    after = sum(p.numel() for p in transformer.parameters())
    assert before > 0
    assert after == 0
    assert controller.dropped_parameter_numel == before
    controller.ensure_key(("score", 1), pairs[4], persistent=False)
    with controller.scope(("score", 1), persistent=False):
        values = transformer.transformer_blocks[0].adaln_proj(torch.zeros(1))
    assert len(values) == 6
