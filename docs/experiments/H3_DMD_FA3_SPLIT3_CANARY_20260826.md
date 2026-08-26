# Iteration 398: FA3 split3 production-shape canary

At the exact BF16 attention shape `[1,37760,56,128]`, split3 passed the existing
output/gradient numerical gate but was slower and used more temporary HBM than
the current split2 path.

| Measurement | split2 | split3 |
|---|---:|---:|
| no-grad forward median | 183.060 ms | 184.271 ms |
| grad-enabled forward median | 183.290 ms | 184.864 ms |
| forward + backward median | 692.469 ms | 694.262 ms |
| canary peak allocated | 6.133 GiB | 8.150 GiB |

Using the live 1,300 no-grad plus 624 grad/replay call weights, split3 projects
**2.556 seconds/GPU/cycle slower**. No training source or world16 workload was
run. Split2 remains the selected FA3 setting.

