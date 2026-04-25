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

For larger sample sizes:

```bash
python benchmarks/timing.py --all --n-samples 10000 --n-rows 10000 --n-spectra 10000
```

Results are auto-printed as a markdown table and can be saved to JSON with `--output results.json`.

## Inference Timings (Default Sample Sizes)

Default: 1000 rows for broadband, 1000 spectra for VAE, 100 samples for LDM.

| Operation | NERSC A100 First (ms) | NERSC A100 Avg (ms) | NERSC CPU First (ms) | NERSC CPU Avg (ms) | Entropy 3090 First (ms) | Entropy 3090 Avg (ms) | Entropy CPU First (ms) | Entropy CPU Avg (ms) |
|---|---|---|---|---|---|---|---|---|
| Broadband MLP (1000 rows) | — | — | 85.01 | 0.97 | 856.51 | 0.06 | 75.84 | 1.12 |
| VAE encode+decode (1000) | — | — | 834.05 | 53.44 | 2894.82 | 1.80 | 921.05 | 28.14 |
| LDM dark (100, deterministic) | — | — | 4380.27 | 1928.83 | 6594.93 | 130.10 | 6611.28 | 4497.94 |
| LDM moon (100, deterministic) | — | — | 3722.56 | 2162.67 | 3687.77 | 127.74 | 5903.29 | 4502.59 |
| LDM twilight (100, deterministic) | — | — | 4056.43 | 2387.14 | 3581.42 | 128.66 | 6102.93 | 4506.11 |

> **Note:** NERSC A100 default-size inference results are pending. The 10K results below are more representative of real workloads.

## Inference Timings (10K Sample Sizes)

10,000 rows for broadband, 10,000 spectra for VAE, 10,000 samples for LDM.

| Operation | NERSC A100 First (ms) | NERSC A100 Avg (ms) | NERSC CPU First (ms) | NERSC CPU Avg (ms) |
|---|---|---|---|---|
| Broadband MLP (10000 rows) | 693.52 | 0.15 | 79.44 | 2.90 |
| VAE encode+decode (10000) | 5853.44 | 11.08 | 3113.28 | 1291.51 |
| LDM dark (10000, deterministic) | 80944.01 | 1043.79 | 30406.20 | 27227.42 |
| LDM moon (10000, deterministic) | 5454.53 | 1043.55 | 29378.19 | 26945.15 |
| LDM twilight (10000, deterministic) | 5647.12 | 1042.45 | 33998.50 | 27081.16 |

### Key Takeaways (10K samples)

- **LDM GPU speedup**: ~26x for 10K samples (1,043ms vs 27,227ms avg for dark deterministic)
- **VAE GPU speedup**: ~117x (11ms vs 1,292ms avg)
- **Broadband GPU speedup**: ~19x (0.15ms vs 2.90ms avg)
- GPU first-eval overhead is high for LDM dark deterministic (81s) due to JIT compilation, but subsequent evals are fast

## Training Timings

| Operation | NERSC A100 First (ms) | NERSC A100 Avg (ms) | NERSC CPU First (ms) | NERSC CPU Avg (ms) | Entropy 3090 First (ms) | Entropy 3090 Avg (ms) | Entropy CPU First (ms) | Entropy CPU Avg (ms) |
|---|---|---|---|---|---|---|---|---|
| Broadband (10 epochs) | 2022.72 | 67.12 | — | — | 1591.20 | 66.00 | 380.97 | 151.83 |
| VAE (10 epochs) | — | — | — | — | 6942.24 | 211.00 | 2809.96 | 2032.58 |
| LDM dark (10 epochs) | 36212.82 | 522.40 | — | — | 30812.33 | 846.30 | 10369.17 | 1945.42 |

> **Note:** NERSC VAE training benchmark is pending — JAX versioning issues on Perlmutter need to be resolved. NERSC CPU training benchmarks are also pending.

## Platform Details

| Platform | Device | JAX Version | Python | Notes |
|---|---|---|---|---|
| NERSC Perlmutter (GPU) | NVIDIA A100-SXM4-40GB | 0.9.1 | 3.11.15 | Shared/exclusive GPU node, 4 devices available |
| NERSC Perlmutter (CPU) | AMD EPYC 7763 | 0.9.1 | 3.11.15 | Exclusive CPU node |
| Entropy (GPU) | NVIDIA RTX 3090 | 0.7.1 | 3.11.13 | |
| Entropy (CPU) | CPU | 0.7.1 | 3.11.13 | |
