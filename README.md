# GloMar Gridding

Library for performing Gridding as used by the GloMar datasets produced by the National Oceanography
Centre. Currently only available to project collaborators.

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
uv sync --python 3.11 # Install dependencies, recommended python version
```

#### `pip` instructions

For development:

```bash
cd /path/to/glomar_gridding
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Acknowledgements

Supported by the Natural Environmental Research Council through National Capability funding
(AtlantiS: NE/Y005589/1)
