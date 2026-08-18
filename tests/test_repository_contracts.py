from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_two_node_config_selects_shared_model_hsdp_and_reorder():
    config = yaml.safe_load(
        (ROOT / "configs/minimax_h3_t2av_dmd_a100_2x8.yaml").read_text(encoding="utf-8")
    )
    assert config["model"]["name"] == "minimax_h3_t2av_a100"
    assert config["model"]["fsdp_load_on_cpu"] is True
    assert config["distributed"]["hybrid_shard"] == {"enabled": True, "shard_size": 8}
    assert config["training"]["method"] == "minimax_h3_t2av_dmd_a100"
    assert config["training"]["a100"]["adaln_cache"]["enabled"] is True
    assert config["training"]["a100"]["critic_rollout_reorder"]["group_size"] == 5
    assert config["training"]["dmd"]["fake_update_ratio"] == 5


def test_upstream_commit_is_pinned():
    commit = (ROOT / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()
    assert len(commit) == 40
    assert commit == "e4ac7ef0122b79ea75b4af429a34f40456b741d4"


def test_shell_scripts_parse():
    for script in sorted((ROOT / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)
