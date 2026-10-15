===============
Getting Started
===============

Installation
------------

Via Pip
=======

GloMarGridding is available on `PyPI <https://pypi.org/project/glomar_gridding/>`_, and can be
installed with `pip` or `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   pip install glomar_gridding

.. code-block:: bash

   uv add glomar_gridding

From Source
===========

Alternatively, you can clone the repository and install using pip (or uv if preferred).

.. code-block:: bash

   git clone https://github.com/NOCSurfaceProcesses/GloMarGridding.git
   cd GloMarGridding
   python -m venv venv
   source venv/bin/activate
   make install

.. code-block:: bash

   git clone https://github.com/NOCSurfaceProcesses/GloMarGridding.git
   cd GloMarGridding
   uv sync --all-extras --python 3.11
