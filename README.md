# GloMar Gridding

Library for performing Gridding as used by the GloMar datasets produced by the National Oceanography
Centre.

Part of the NOC Surface Processes _GloMar_ suite of libraries and datasets.

## Installation

`GloMarGridding` is available on [PyPI](https://pypi.org/project/glomar_gridding/), as
`glomar_gridding`:

```bash
pip install glomar_gridding
```

Or using uv:

```bash
uv add glomar_gridding
```

### Development

Clone the repository

```bash
git clone https://github.com/NOCSurfaceProcesses/GloMarGridding.git /path/to/glomar_gridding
```

Create virtual environment and install dependencies. We recommend using
[uv](https://docs.astral.sh/uv/) for python as an alternative to `pip`.

```bash
cd /path/to/glomar_gridding
uv sync --all-extras --python 3.11  # Install dependencies, recommended python version
```

#### `pip` instructions

For development:

```bash
cd /path/to/glomar_gridding
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Documentation

Documentation PDF can be found in the docs directory. Or
[here](https://github.com/NOCSurfaceProcesses/GloMarGridding/blob/main/docs/Documentation.pdf).

An example workflow can be found in the documentation PDF, or in the notebooks directory.

## Citation

Richard C. Cornes, Steven. C. Chan, Archie Cable et al. GloMarGridding: A Python Package for Spatial
Interpolation to Support Structural Uncertainty Assessment of Climate Datasets, 22 August 2025,
PREPRINT (Version 1) available at Research Square
[https://doi.org/10.21203/rs.3.rs-7427869/v1](https://doi.org/10.21203/rs.3.rs-7427869/v1)

## Acknowledgements

Supported by the Natural Environmental Research Council through National Capability funding
(AtlantiS: NE/Y005589/1)
