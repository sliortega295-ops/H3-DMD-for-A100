from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_state_cache_is_explicit_and_fail_visible():
    model = (ROOT / "h3_a100" / "model.py").read_text()
    assert "self._physical_role: str | None = None" in model
    assert "if role == self._physical_role:" in model
    assert 'self._runtime_state_stats["role_noops"] += 1' in model
    assert "def set_transformer_training" in model
    assert "self.denoiser_module().train(training)" in model
    assert "def runtime_state_stats" in model


def test_fake_forward_and_replay_share_one_outer_role_scope():
    loop = (ROOT / "h3_a100" / "trainer_loop.py").read_text()
    outer = "with self.shared_model.role_scope(FAKE_ADAPTER):"
    activation = 'with self._activation_offload_scope(f"fake_{micro_idx}") as offload:'
    assert loop.index(outer, loop.index("def _apply_one_fake_group")) < loop.index(
        activation, loop.index("def _apply_one_fake_group")
    )
    assert "[h3-a100][runtime-state]" in loop


def test_hot_path_uses_cached_training_transition():
    runtime = (ROOT / "h3_a100" / "trainer_runtime.py").read_text()
    loop = (ROOT / "h3_a100" / "trainer_loop.py").read_text()
    assert "self.shared_model.transformer.train()" not in runtime
    assert "self.shared_model.transformer.eval()" not in runtime
    assert "self.shared_model.transformer.train()" not in loop
    assert "self.shared_model.set_transformer_training(True)" in runtime
    assert "self.shared_model.set_transformer_training(False)" in runtime
