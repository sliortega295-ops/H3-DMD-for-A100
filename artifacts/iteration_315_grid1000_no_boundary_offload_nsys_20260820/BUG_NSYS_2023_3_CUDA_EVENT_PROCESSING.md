# Nsight Systems 2023.3.1 capture bug

Run `iteration_315_grid1000_no_boundary_offload_nsys_20260820` completed the actual
world16 Grid1000 one-cycle workload, but the Nsight 2023.3.1 report postprocessor emitted:

- `Unknown runtime API function index: 468`
- `Cannot find bucket for a bucket index`
- `Cannot find string for an exterior index`

Consequences:

- node0 exported kernels only for CUDA devices 2 and 7;
- node1 exported only 20 kernels on device 4 and missed `h3/critic_update_F`;
- most kernel names became `UNRESOLVED_KERNEL`/`QnxKernelTrace`;
- no full-world compute or NCCL exposed-time result is admissible.

The workload itself passed its application census and Grid replay gates. `copyKind=2`
(D2H) in the workload range was 0 B on node0 and 8 B on node1, consistent with
`activation_policy=none` and `boundary_events=False`; this is not an offload failure.

Recommended fix: rerun with a newer Nsight Systems build compatible with the CUDA/PyTorch
runtime, or disable the unsupported runtime-event path while retaining CUDA kernel, NVTX,
and memcpy capture. Do not use the partial profile for timing or speedup.
