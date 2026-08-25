import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_source(name: str) -> str:
    source = (ROOT / "h3_a100" / "trainer_loop.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


def test_hot_training_steps_do_not_read_device_scalars():
    for name in ("_student_step", "_fake_steps", "_apply_one_fake_group"):
        assert ".item()" not in _function_source(name)


def test_scalar_materialization_is_after_cycle_receipts():
    source = (ROOT / "h3_a100" / "trainer_loop.py").read_text()
    assert source.index("self._validate_fused_pointwise_cycle(current_iter)") < source.index(
        "display_dmd = reduce_mean(self._mean_detached_metrics(running_dmd))"
    )
    assert "return torch.stack(values).mean()" in source
