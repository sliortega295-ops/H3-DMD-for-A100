from __future__ import annotations

import json

import torch

from h3_a100 import grid_timestep_compat as _compat  # noqa: F401
from h3_a100.grid_adaln import GridAdaLNController, SCHEMA_VERSION


def _bits(values: torch.Tensor):
    return [int(v) for v in values.to(torch.float32).contiguous().view(torch.int32).tolist()]


def test_single_runtime_timestep_maps_to_duplicated_fixed_table_row(tmp_path):
    hidden = 1
    shape = (1, 50, 6, 6, hidden)
    numel = 1
    for value in shape:
        numel *= value
    binary = tmp_path / "table.bin"
    table = torch.from_file(str(binary), shared=True, size=numel, dtype=torch.bfloat16).reshape(shape)
    table.fill_(3.0)
    del table

    timestep = torch.tensor([0.0], dtype=torch.float32)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "binary_file": binary.name,
        "grid_size": 1,
        "num_rollout_entries": 1,
        "num_entries": 1,
        "num_blocks": 50,
        "hidden_size": hidden,
        "rows_per_entry": 6,
        "modulation_chunks": 6,
        "dtype": "torch.bfloat16",
        "timestep_bits": [_bits(torch.tensor([0.0, 0.0], dtype=torch.float32))],
    }
    path = tmp_path / "table.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    controller = GridAdaLNController(path, pin_memory=False)
    controller.ensure_key(("rollout", 0), timestep, persistent=True)
    with controller.scope(("rollout", 0), persistent=True):
        chunks = controller.lookup(0)
    assert len(chunks) == 6
    assert torch.all(chunks[0] == torch.tensor(3.0, dtype=torch.bfloat16))
