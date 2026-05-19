# Task-specific programming of chaos in neural circuits

This repository contains code to reproduce the manuscript figures and supplementary figures.
- `Fig1/`: self-contained figure generation.
- `Fig2/`: `Fig2ab` requires a precomputed sweep; `Fig2cd` is self-contained.
- `Fig3/`: `Fig3abc` requires a precomputed sweep; `Fig3e` is self-contained.
- `Fig4/`: `Fig4a` is self-contained; `Fig4bcd` requires training result outputs generated first.
- `supp/`: supplementary figure plotting requires sweeping outputs generated first, and `FigS4` also depends on the main `Fig3` sweep output.
- `utils/`: contains necessary utility functions, including small-world network generation  and GPU-optimized solvers for the main ODE (`utils/swneural.py`), calculation of temporal metrics (`utils/quantities.py`), LLE calculations (`utils/lyapunov.py`), rewiring algorithms (`utils/rewire.py`), and expressivity calculations (`utils/expressivity.py`).

---

## Project structure

```text
SW_Chaos/
├── Fig1/
│   ├── Fig1_creator.py
│   └── results/
├── Fig2/
│   ├── Fig2ab_creator.py
│   ├── Fig2cd_creator.py
│   ├── outputs/
│   │   └── beta_perturb_sweep.npz
│   ├── results/
│   └── scripts/
│       └── sweep_beta_optimized.py
├── Fig3/
│   ├── Fig3abc_creator.py
│   ├── Fig3e_creator.py
│   ├── outputs/
│   │   ├── 2D_perturb_sweep.npz
│   │   └── lyapunov_sweep.npz
│   ├── results/
│   └── scripts/
│       ├── sweep_2D_optimized.py
│       └── sweep_lyapunov.py
├── Fig4/
│   ├── Fig4a_creator.py
│   ├── Fig4bcd_creator.py
│   ├── outputs/
│   │   ├── lorenz_sweep_seed*.npz
│   │   ├── mnist_sweep_seed*.npz
│   │   └── remote_bandwidth_sweep_seed*.npz
│   ├── results/
│   └── scripts/
│       ├── sweep_train.sh
│       ├── sweep_train_lorenz.py
│       ├── sweep_train_mnist.py
│       ├── sweep_train_remote_bandwidth.py
│       ├── train_lorenz.py
│       ├── train_mnist.py
│       └── train_remote_bandwidth.py
├── supp/
│   ├── FigS1_creator.py
│   ├── FigS2S3_creator.py
│   ├── FigS4_creator.py
│   ├── outputs/
│   │   ├── beta_perturb_sweep_eps*.npz
│   │   ├── 2D_perturb_sweep_n*.npz
│   │   └── expressivity_sweep.npz
│   ├── results/
│   └── scripts/
│       ├── sweep_2D_optimized_supp.py
│       ├── sweep_beta_optimized_supp.py
│       ├── sweep_eps.sh
│       ├── sweep_expressivity.py
│       └── sweep_n.sh
├── utils/
├── pyproject.toml
└── README.md
```

---

## Requirements

Recommended: use `uv`. From the repository root:

```bash
uv sync
```

If you prefer `pip` instead of `uv`:

```bash
pip install -r requirements.txt
```

When the CUDA version is not 12, check the current CUDA version of your device, then run (instead of default `jax[cuda12]`):

```bash
# CUDA 13
pip install -U "jax[cuda13]"

# CPU-only
pip install -U jax
```

If you installed with `pip`, run commands directly (without `uv run`), e.g.:
- `python -m Fig1.Fig1_creator`
- `bash ./Fig4/scripts/sweep_train.sh`

Main dependencies used across the repo:
- `jax[cuda12]`, `diffrax`
- `numpy`, `matplotlib`
- `networkx`
- `torch`, `torchvision`: required for MNIST training
- `tqdm`

GPU note:
- This project explicitly depends on `jax[cuda12]` (see `pyproject.toml`), so NVIDIA GPU runs expect a CUDA 12 setup with compatible cuDNN.
- If your machine is configured for CUDA 13, `jax[cuda13]` is also expected to work for this codebase.
- CPU-only execution is still possible, but heavy sweep/training workloads will be much slower.
- Quick smoke test after install: `uv run python -c "import jax, diffrax; print(jax.devices())"`.

---

## How to reproduce figures

All commands below assume the current working directory is the repository root. All scripts depend on the utility functions defined in `utils/*.py`.

### Fig.1 (self-contained)
Run:

```bash
uv run python -m Fig1.Fig1_creator
```

Produces:
- `Fig1/results/Fig1b.tiff`
- `Fig1/results/Fig1c.tiff`
- `Fig1/results/Fig1d.tiff`

---

### Fig.2 (requires data generation first)
**Data generation**

Run:

```bash
uv run python -m Fig2.scripts.sweep_beta_optimized
```

This writes:
- `Fig2/outputs/beta_perturb_sweep.npz`

Notes:
- The script defaults to parallelizing the samples across all available GPUs. Adjust the number of available GPUs by adjusting the environment variable `CUDA_VISIBLE_DEVICES`.

**Figure generation**

Run:

```bash
uv run python -m Fig2.Fig2ab_creator
uv run python -m Fig2.Fig2cd_creator
```

Produces:
- `Fig2/results/Fig2a.tiff`
- `Fig2/results/Fig2b.tiff`
- `Fig2/results/Fig2c.tiff`
- `Fig2/results/Fig2d.tiff`

Notes:
- `Fig2ab_creator.py` loads `Fig2/outputs/beta_perturb_sweep.npz`.
- `Fig2cd_creator.py` is self-contained.

---

### Fig.3 (requires data generation first)
**Data generation**

Run:

```bash
uv run python -m Fig3.scripts.sweep_2D_optimized
uv run python -m Fig3.scripts.sweep_lyapunov
```

These scripts write:
- `Fig3/outputs/2D_perturb_sweep.npz`
- `Fig3/outputs/lyapunov_sweep.npz`

Notes:
- The script defaults to parallelizing the samples across all available GPUs. Adjust the number of available GPUs by adjusting the environment variable `CUDA_VISIBLE_DEVICES`.

**Figure generation**

Run:

```bash
uv run python -m Fig3.Fig3abc_creator
uv run python -m Fig3.Fig3e_creator
```

Produces:
- `Fig3/results/Fig3a.tiff`
- `Fig3/results/Fig3b.tiff`
- `Fig3/results/Fig3c.tiff`
- `Fig3/results/Fig3e.tiff`

Notes:
- `Fig3abc_creator.py` loads both `.npz` files above.
- `Fig3e_creator.py` is self-contained.

---

### Fig.4 (requires training/sweep outputs for Fig.4b-d)
**Data generation**

Run:

```bash
uv run bash ./Fig4/scripts/sweep_train.sh
```

This shell script runs the three training/sweep jobs across multiple seeds:
- Lorenz forecasting
- MNIST classification
- Remote signal tracking

It writes files such as:
- `Fig4/outputs/lorenz_sweep_seed*.npz`
- `Fig4/outputs/mnist_sweep_seed*.npz`
- `Fig4/outputs/remote_bandwidth_sweep_seed*.npz`

Notes:
- The MNIST run will generate a `./data` directory, containing the MNIST image datasets. This requires the MNIST data to be downloadable, i.e., with internet connection or with `./data/MNIST` to be already installed.

**Figure generation**

Run:

```bash
uv run python -m Fig4.Fig4a_creator
uv run python -m Fig4.Fig4bcd_creator
```

Produces the main outputs:
- `Fig4/results/Fig4a_lorenz.tiff`
- `Fig4/results/Fig4a_lorenz_2.tiff`
- `Fig4/results/Fig4a_mnist_*.tiff`: MNIST samples
- `Fig4/results/Fig4a_remote_bandwidth.tiff`
- `Fig4/results/Fig4a_remote_bandwidth_2.tiff`
- `Fig4/results/Fig4b.tiff`
- `Fig4/results/Fig4c.tiff`
- `Fig4/results/Fig4d.tiff`

Notes:
- `Fig4a_creator.py` requires the MNIST data to be downloadable, i.e., with internet connection or with `./data/MNIST` to be already installed.
- `Fig4bcd_creator.py` loads the `.npz` files in `Fig4/outputs/`.

---

### Supplementary figures (requires data generation first)
**Data generation**

Run:

```bash
uv run bash ./supp/scripts/sweep_eps.sh
uv run bash ./supp/scripts/sweep_n.sh
uv run python -m supp.scripts.sweep_expressivity
```

These commands write:
- `supp/outputs/beta_perturb_sweep_eps*.npz`
- `supp/outputs/2D_perturb_sweep_n*.npz`
- `supp/outputs/expressivity_sweep.npz`

**Figure generation**

Run:

```bash
uv run python -m supp.FigS1_creator
uv run python -m supp.FigS2S3_creator
uv run python -m supp.FigS4_creator
```

Produces:
- `supp/results/FigS1_eps*.tiff`
- `supp/results/FigS2S3_tau_c_N*.tiff`
- `supp/results/FigS2S3_tau_p_N*.tiff`
- `supp/results/FigS4a.tiff`
- `supp/results/FigS4b.tiff`

Notes:
- `FigS1_creator.py` loads the `beta_perturb_sweep_eps*.npz` files.
- `FigS2S3_creator.py` loads the `2D_perturb_sweep_n*.npz` files.
- `FigS4_creator.py` loads `supp/outputs/expressivity_sweep.npz` and `Fig3/outputs/2D_perturb_sweep.npz`, so the main Fig.3 sweep must also be generated first.

---

## Tips

- Run every command from the repository root.
- Use `uv run ...` for both Python scripts and shell scripts. This ensures the commands execute inside the project environment.
- The data generation sweeps are substantially heavier than the self-contained figure scripts, and running on a GPU machine is recommended.
- If you are on a machine without a configured CUDA setup, JAX may fall back to CPU or emit CUDA-related warnings depending on your environment.

---

## Output files

Generated data files are saved as `.npz` files inside each figure directory's `outputs/` folder.

Generated figures are saved as `.tiff` files inside each figure directory's `results/` folder, for example:
- `Fig2/results/Fig2a.tiff`
- `Fig3/results/Fig3b.tiff`
- `Fig4/results/Fig4d.tiff`
- `supp/results/FigS4a.tiff`
