# Hyperspectral Crop Disease

Conditional Glow pipeline for generating and evaluating synthetic hyperspectral crop disease images. The project works with 128x128 image cubes with 125 spectral bands and 10 disease-severity classes.

## Repository

- `src/` training, evaluation, dataset, model, and preprocessing code
- `data/` raw class folders plus processed split files and normalization stats
- `notebooks/` exploration, result visualization, and flow-diagnostics notebooks
- `results/` generated samples, evaluation outputs, and report figures
- `docs/` Sphinx documentation generated from source docstrings

## Quick Start

```bash
uv sync
```

Train the flow:

```bash
uv run python -m src.train
```

Evaluate a checkpoint:

```bash
uv run python -m src.evaluate --checkpoint results/flow/epoch_125.pt
```

Notebooks:

```bash
uv run jupyter lab
```

## Documentation

Install the development dependencies and build the Sphinx site:

```bash
uv sync --group dev
uv run sphinx-build -b html docs/source docs/build/html
```

The generated HTML documentation is written to `docs/build/html/index.html`.
