# Iteration 316: Nsight upgrade preflight

Status: **FAIL_NSYS_VERSION**. No sanity CUDA test and no world16 profiling were run.

The exact source was reset to `04b3a2eb0a1b3bbbc3d83575d4059d43804fcbd0` on
`codex/grid1000-adaln-results-20260820`. Both owned nodes resolve:

```text
/usr/local/cuda/bin/nsys
 -> /usr/local/cuda-12.2/NsightSystems-cli-2023.3.1/target-linux-x64/nsys
NVIDIA Nsight Systems version 2023.3.1.92-233133147223v0
```

The requested minimum is Nsight Systems >= 2024.6.2. No alternate >=2024.6.2
binary was found in the bounded `/usr/local`, `/opt`, `/home/shuchenweng`, and
`/share/project/shuchenweng` searches. GPU ownership was clear during the check:
all 8 A100-SXM4-40GB per node had no compute applications. Existing cgroup counters
were recorded but not modified.

Per the run contract, profiling stopped here to avoid another invalid 15-minute
world16 run. The previous parser-failure evidence remains in iteration 315.

The analyzer was nevertheless corrected statically in the parent artifact:
repeated NVTX ranges are stored as lists, critic occurrences are not overwritten,
and metrics are calculated per node/device/phase with a fail-closed validity gate.
This change was compile-checked and exercised against the old invalid SQLite only;
it does not make that profile valid.

Next required external change: provide a user-accessible Nsight Systems >=2024.6.2
on both nodes (or an approved compatible container/tool path), then rerun the 30-second
sanity capture before launching world16.
