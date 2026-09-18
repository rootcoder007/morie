Deployment
==========

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE supports deployment across consumer hardware, single-board computers,
and containers. The target is a single-command install that auto-detects
and configures all dependencies.

From source
-----------

.. code-block:: bash

   git clone https://github.com/rootcoder007/morie.git
   cd morie
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[interactive]"

The package is pure Python at runtime with four small dependencies
(``openpyxl``, ``httpx``, ``rich``, ``beautifulsoup4``); nothing has to
compile for your architecture. ``morie doctor`` reports the state of the
optional pieces (R, Ollama, Docker, the Terminal IDE).

Docker
------

MORIE provides a multi-architecture Docker image for inspection, testing,
and verification.

**Dockerfile highlights:**

- Base image pinned by digest (not tag) for reproducibility
- Non-root user ``morieapp`` (UID 1000)
- Read-only root filesystem with ``/tmp`` writable
- Health check via ``morie selftest``
- Layer ordering: system deps, Python deps, then source (cache-friendly)

.. code-block:: bash

   docker build -t morie:latest .
   docker run --rm morie:latest morie selftest
   docker run --rm morie:latest morie verify
   docker run --rm -it morie:latest morie repl

**Multi-arch build (amd64 + arm64):**

.. code-block:: bash

   docker buildx build --platform linux/amd64,linux/arm64 \
     -t ghcr.io/rootcoder007/morie:latest --push .

.. note::

   The published image is ``ghcr.io/rootcoder007/morie``, rebuilt by CI on
   every push to ``main``; ``docker pull`` needs no authentication.

Ollama Sidecar
--------------

For LLM-enabled deployments, Ollama runs as a sidecar container:

.. code-block:: yaml

   # docker-compose.yml
   services:
     morie:
       image: morie:latest
       environment:
         - OLLAMA_HOST=ollama:11434
       depends_on:
         - ollama
     ollama:
       image: ollama/ollama:latest
       volumes:
         - ollama_data:/root/.ollama
       ports:
         - "11434:11434"
   volumes:
     ollama_data:

MORIE auto-detects the Ollama sidecar via the ``OLLAMA_HOST`` environment
variable. No additional configuration is needed.

systemd Service
---------------

For persistent deployments (e.g., Pi running as a headless analysis server):

.. code-block:: ini

   [Unit]
   Description=MORIE Analysis Server
   After=network.target ollama.service

   [Service]
   Type=simple
   User=morieapp
   WorkingDirectory=/opt/morie
   ExecStart=/opt/morie/.venv/bin/python -m morie.runner serve
   Restart=on-failure
   RestartSec=5
   Environment=OLLAMA_HOST=localhost:11434

   [Install]
   WantedBy=multi-user.target

Install Methods
---------------

- **pip (PyPI)**: ``pip install morie``.
- **pip (editable)**: ``pip install -e ".[test,docs]"``.
- **brew**: ``brew tap rootcoder007/morie && brew install morie``.
- **curl installer**: ``curl -fsSL https://rootcoder007.github.io/morie/install.sh | bash``
  detects the platform (macOS/Linux/WSL), installs Python and R as needed,
  then both packages.
- **Docker**: ``docker pull ghcr.io/rootcoder007/morie:latest``.
- **R**: ``install.packages("rmorie", repos = "https://rootcoder007.r-universe.dev")``.

Cross-Platform Notes
--------------------

- **macOS (Apple Silicon)**: primary development platform; CRAN R or
  Homebrew R both work.
- **Linux (x86_64)**: fully supported. Docker is the recommended
  deployment method for production.
- **Linux (arm64)**: supported, including Raspberry Pi. Ollama ARM64
  builds work natively.
- **Windows**: native Windows is checked in CI for the R package and works
  for the Python package; WSL2 with Ubuntu also works.
