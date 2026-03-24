# ST_learn_synthesis — Learning Spatio-Temporal Specifications from Biological Cell Simulations

A research pipeline for learning spatio-temporal logic (STL) formulas that describe the collective behavior of biological cell populations. The system generates simulation data from agent-based models (ABMs) or reaction-diffusion systems, clusters the resulting trajectories, and trains PyTorch models to learn the governing behavioral rules.

## Motivation

Formal specifications (e.g., signal temporal logic formulas) can compactly describe *when* and *where* certain behaviors occur in a system. This project asks: given a set of simulation trajectories, can we automatically learn the STL formula that best characterizes the observed behavior? The answer has applications in biological system identification, model validation, and spec-guided control.

## Pipeline

```
1. Generate_data.py      — run simulations (ABM / Turing / Morpheus), save trajectories as .pkl
        │
        ▼
2. Label_data.py         — cluster image frames and trajectories (KMeans / TimeSeriesKMeans)
        │
        ▼
3. Learning_formulae.py  — train PyTorch model on labeled trajectory data to learn STL formulas
```

## Simulation Backends

Three simulation types are supported, selected via `model_name` in `Generate_data.py`:

### `abm` — Agent-Based Model of Cell Populations (`abm/`)
A custom 3D agent-based simulator with three cell types (blue, red, yellow). Cells interact via:
- **Adhesion forces** — type-specific spring-like attraction between neighboring cells (e.g., yellow-yellow adhesion `u_yy=30` is stronger than blue-blue `u_bb=5`)
- **Short-range repulsion** — prevents cell overlap
- **Gravity well confinement** — keeps the population within a bounded region

Physics are computed with Numba JIT-compiled kernels for performance. Simulations are configured via YAML parameter files in `abm/yaml_parameters/` and output images + videos to `abm/outputs/`.

### `turing` — Turing Reaction-Diffusion Patterns
Simulates a two-component Turing system (`A`, `B`) on a 2D grid. Produces spatial pattern trajectories saved as `.pkl` files.

### `morpheus` — Morpheus Cell Simulation
Wraps the [Morpheus](https://morpheus.gitlab.io/) multicellular modeling platform via `abm/MorpheusSetup.py`. Runs via subprocess with chemotaxis strength parameters swept across a grid.

## Project Structure

| File | Description |
|---|---|
| `Generate_data.py` | Runs simulations in parallel (`multiprocessing.Pool`); saves output trajectories as `.pkl` |
| `Label_data.py` | Loads trajectories/videos, clusters images (`clustimage`) and time series (`TimeSeriesKMeans`), saves labeled data |
| `Learning_formulae.py` | Loads labeled data, trains a PyTorch model to learn STL formulas, supports `abm`, `turing1`, `turing2`, `rand` cases |
| `Optimization.py` | Formula optimization utilities |
| `Write_formula.py` | Serializes/formats learned formulas |
| `abm/model.py` | `TestSimulation` — defines cell types, forces, division/death rules for the ABM |
| `abm/simulation.py` | Abstract `Simulation` base class — handles stepping, output recording, image/video generation |
| `abm/backend.py` | Low-level utilities: neighbor search (CPU/GPU), graph construction, bin assignment |
| `abm/parameter_sweep.py` | Sweeps YAML parameter files to generate a batch of ABM runs |
| `abm/make_yaml.py` | Generates YAML config files for parameter sweeps |
| `abm/display_images.py` | Visualizes simulation output images |

## Running

### 1. Generate simulation data

```bash
# Edit model_name in Generate_data.py ("abm", "turing", or "morpheus"), then:
python Generate_data.py
```

Output trajectories are saved as `.pkl` files to the path set in the script (default: `D:/Projects/ST_learn_synthesis/output/`). Update this path to match your local setup.

### 2. Label the data

```bash
python Label_data.py
```

Clusters image frames and temporal trajectories. Saves labeled arrays (`*_img_y.pkl`, `*_img_X.pkl`, etc.) to the output directory.

### 3. Train the model

```bash
# Set `case` in Learning_formulae.py to match your data ("abm", "turing1", "turing2", or "rand")
python Learning_formulae.py
```

Uses GPU if available (`torch.cuda`).

### Running the ABM directly

```python
from abm import model
model.TestSimulation.start_sweep('outputs/', 'my_params.yaml', 'my_run', seed=0)
```

## Requirements

```bash
pip install numpy torch torchvision matplotlib pandas scikit-learn opencv-python numba psutil clustimage tslearn mat73
```

GPU support requires a CUDA-compatible device and the appropriate PyTorch CUDA build.

## Notes

- Output paths are currently hardcoded as Windows absolute paths (e.g., `D:/Projects/...`). Update these in `Generate_data.py`, `Label_data.py`, and `Learning_formulae.py` before running.
- The `abm/yaml_parameters/` directory contains pre-generated sweep configs for ~1000+ ABM runs.
- Simulation outputs (images, videos) are in `abm/outputs/`.

## Related Work

- [`MR_RL`](https://github.com/SuhailSama/MR_RL) — applies RL to a related micro-robot control problem using similar Gym-style simulation infrastructure
- [`DQN4MRs`](https://github.com/SuhailSama/DQN4MRs) — includes `STREL_RL.py`, an extension that uses signal temporal logic to shape RL reward signals
