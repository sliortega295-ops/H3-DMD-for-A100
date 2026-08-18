from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_primary_config_is_world16_matched_control():
    config = yaml.safe_load(
        (ROOT / "configs/minimax_h3_t2av_dmd_a100_world16.yaml").read_text(encoding="utf-8")
    )
    assert config["model"]["name"] == "minimax_h3_t2av_a100"
    assert config["model"]["fsdp_load_on_cpu"] is True
    assert config["distributed"]["hybrid_shard"]["enabled"] is False
    assert config["distributed"]["fsdp2"]["size"] == 16
    assert config["training"]["a100"]["matched_compute"]["fixed_end_step_idx"] == 3
    assert config["training"]["a100"]["matched_compute"]["expected_world_size"] == 16
    assert config["training"]["a100"]["adaln_cache"]["enabled"] is True
    assert config["training"]["a100"]["critic_rollout_reorder"]["group_size"] == 5
    assert config["training"]["dmd"]["fake_update_ratio"] == 5


def test_hsdp_config_keeps_same_compute_contract():
    config = yaml.safe_load(
        (ROOT / "configs/minimax_h3_t2av_dmd_a100_2x8.yaml").read_text(encoding="utf-8")
    )
    assert config["distributed"]["hybrid_shard"] == {"enabled": True, "shard_size": 8}
    assert config["training"]["a100"]["matched_compute"]["fixed_end_step_idx"] == 3
    assert config["training"]["a100"]["matched_compute"]["expected_world_size"] == 16
    assert config["data"]["train"]["batch_size"] == 1


def test_upstream_commit_matches_dmd_system_reference():
    commit = (ROOT / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()
    assert len(commit) == 40
    assert commit == "d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be"


def test_shell_scripts_parse():
    for script in sorted((ROOT / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)
