# DiffLiB

![Github Star](https://img.shields.io/github/stars/xwpken/DiffLiB) ![Github Fork](https://img.shields.io/github/forks/xwpken/DiffLiB) ![License](https://img.shields.io/github/license/xwpken/DiffLiB.svg)

Differentiable Lithium-ion batteries simulation framework.

## Installation

First intsall `JAX` (see [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)), then follow the [JAX-FEM instructions](https://github.com/deepmodeling/jax-fem?tab=readme-ov-file#installation) to install `JAX-FEM`, which will create a new conda environment.

Activate the environment and clone the repository:

```bash
git clone https://github.com/xwpken/DiffLiB.git
cd DiffLiB
```

then install the package locally:

```bash
pip install -e .
```

## Quick start

Forward predictions:

```bash
python -m examples.forward.main
```
Gradient computations:

```bash
python -m examples.gradient.main
```
## Citations

If you found this library useful, we appreciate your support if you consider citing the following paper:

```bibtex
to be added
```