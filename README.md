# GloMar Gridding

Library for performing Gridding as used by the GloMar datasets produced by the National Oceanography Centre.
Currently only available to project collaborators.

Part of the NOC Surface Processes _GloMar_ suite of libraries and datasets.

## Installation

Clone the repository

```bash
git clone https://github.com/NOCSurfaceProcesses/GloMarGridding.git /path/to/glomar_gridding
```

Create virtual environment and install dependencies. We recommend using [`uv`](https://docs.astral.sh/uv/) for python as an alternative to `pip`.

```bash
cd /path/to/glomar_gridding
uv venv --python 3.11  # Recommended python version
source .venv/bin/activate  # Assuming bash or zsh
uv sync  # Install dependencies
```

### Install as a dependency

```bash
uv add git+https://github.com/NOCSurfaceProcesses/GloMarGridding.git
```

### `pip` instructions

For development:

```bash
cd /path/to/glomar_gridding
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or as a dependency:

```bash
pip install git+https://github.com/NOCSurfaceProcesses/GloMarGridding.git
```

## Acknowledgements

Supported by the Natural Environmental Research Council through National Capability funding
(AtlantiS: NE/Y005589/1)
