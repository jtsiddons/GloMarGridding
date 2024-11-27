# NOC Kriging

Library for performing kriging. Currently only available to project collaborators.

## Installation

Clone the repository

```bash
git clone git@git.noc.ac.uk:nocsurfaceprocesses/kriging_split_into_modules.git /path/to/noc_kriging
```

Create virtual environment and install dependencies. We recommend using [`uv`](https://docs.astral.sh/uv/) for python as an alternative to `pip`.

```bash
cd /path/to/noc_kriging
uv venv --python 3.11
source .venv/bin/activate  # Assuming bash or zsh
uv sync
```

### Install as a dependency

```bash
uv add git+git@git.noc.ac.uk/nocsurfaceprocesses/kriging_split_into_modules.git
```

### `pip` instructions

For development:

```bash
cd /path/to/noc_kriging
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or as a dependency:

```bash
pip install git+git@git.noc.ac.uk/nocsurfaceprocesses/kriging_split_into_modules.git
```
