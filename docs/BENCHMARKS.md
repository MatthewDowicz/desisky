# Benchmarks

This document tracks benchmark results for `desisky` across different machines and computation backends (JAX CPU, JAX GPU).

We report two timings for each operation:

- **First eval** — first call (includes JAX JIT compilation overhead)
- **Avg eval** — average over N repeated calls (post-JIT steady-state)

**Benchmark procedure:** The timings below are obtained by running the benchmark script in the repository:

```bash
python benchmarks/timing.py --all
```

To run only inference or training benchmarks:

```bash
python benchmarks/timing.py --inference
python benchmarks/timing.py --training
```

Results are auto-printed as a markdown table and can be saved to JSON with `--output results.json`.

## Inference Timings

| Operation | NERSC A100 First (ms) | NERSC A100 Avg (ms) | NERSC CPU First (ms) | NERSC CPU Avg (ms) | Entropy 3090 First (ms) | Entropy 3090 Avg (ms) | Entropy CPU First (ms) | Entropy CPU Avg (ms) |
|---|---|---|---|---|---|---|---|---|
| Broadband MLP (1000 rows) | | | | | 856.51 | 0.06 | 75.84 | 1.12 |
| VAE encode+decode (1000) | | | | | 2894.82 | 1.80 | 921.05 | 28.14 |
| LDM dark (100, deterministic) | | | | | 6594.93 | 130.10 | 6611.28 | 4497.94 |
| LDM dark (100, stochastic) | | | | | 3986.28 | 128.12 | 5788.69 | 4282.69 |
| LDM moon (100, deterministic) | | | | | 3687.77 | 127.74 | 5903.29 | 4502.59 |
| LDM twilight (100, deterministic) | | | | | 3581.42 | 128.66 | 6102.93 | 4506.11 |

## Training Timings

| Operation | NERSC A100 First (ms) | NERSC A100 Avg (ms) | NERSC CPU First (ms) | NERSC CPU Avg (ms) | Entropy 3090 First (ms) | Entropy 3090 Avg (ms) | Entropy CPU First (ms) | Entropy CPU Avg (ms) |
|---|---|---|---|---|---|---|---|---|
| Broadband (10 epochs) | | | | | 1591.20 | 66.00 | 380.97 | 151.83 |
| VAE (10 epochs) | | | | | 6942.24 | 211.00 | 2809.96 | 2032.58 |
| LDM dark (10 epochs) | | | | | 30812.33 | 846.30 | 10369.17 | 1945.42 |

## Platform Details

| Platform | Device | JAX Version | Python | Notes |
|---|---|---|---|---|
| NERSC Perlmutter (GPU) | NVIDIA A100 40GB | | | Shared/exclusive GPU node |
| NERSC Perlmutter (CPU) | AMD EPYC 7763 | | | Exclusive CPU node |
| Entropy (GPU) | NVIDIA RTX 3090 | 0.7.1 | 3.11.13 | |
| Entropy (CPU) | CPU | 0.7.1 | 3.11.13 | |
