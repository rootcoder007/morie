Installation
============

MORIE targets Python 3.10 or newer and R 4.3 or newer. The two languages are
independent: install the one you use, or both for the dual-language pipeline.

.. note::
   **About the ``#`` characters in code blocks below.** Lines starting
   with ``#`` (and inline ``# …`` annotations after a command) are
   *comments*. They explain what the command does but are NOT part of the
   command itself. Pasting them into a shell will either error out or run
   something you didn't mean to.

   Hover any code block on this page and click **Copy** in the corner:
   the smart-copy strips ``# comments`` automatically so the result
   is paste-safe. ``Cmd/Ctrl+C`` on a selection inside a code block
   does the same thing.

Python
------

From PyPI:

.. code-block:: bash

   pip install morie                  # the package, 69 built-in datasets, 18,560 morie.fn callables
   pip install "morie[interactive]"   # + the Terminal IDE (textual)

The runtime dependencies are small and pure Python (``openpyxl``, ``httpx``,
``rich``, ``beautifulsoup4``). There is no NumPy, SciPy or pandas
requirement: every estimator runs on MORIE's native array and frame cores.
A pandas DataFrame passed to any function is converted on entry.

The other extras are for development: ``test`` (pytest) and ``docs``
(Sphinx and its plugins).

Editable install from source:

.. code-block:: bash

   git clone https://github.com/rootcoder007/morie.git
   cd morie
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[test,interactive]"

.. note::

   On Raspberry Pi OS and Debian Trixie the system ``/usr/bin/python3``
   (3.13.5) segfaults on import for several scientific wheels. That is a
   Debian packaging bug, not a morie bug. Work around it with ``uv``:

   .. code-block:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh
      uv python install 3.12
      uv venv ~/.venvs/morie --python 3.12
      uv pip install --python ~/.venvs/morie/bin/python morie

R
-

The R side of the family is published as ``rmorie`` (with its companions
``rmoriebricklayer``, the shared C/C++ core, and ``rmoriedata``, the data
corpus). ``rmoriebricklayer`` and ``rmoriedata`` are on CRAN; all three are
served from r-universe:

.. code-block:: r

   install.packages(
     "rmorie",
     repos = c("https://rootcoder007.r-universe.dev",
               "https://cloud.r-project.org")
   )
   library(rmorie)

The copy under ``r-package/morie`` in the repository is the same code under
the package name ``morie``; it is what the R API pages on this site are
built from, and it installs from source with

.. code-block:: r

   install.packages(c("rmoriebricklayer", "rmoriedata"))
   install.packages("r-package/morie", repos = NULL, type = "source")

Either way, every exported function is called ``morie_*``.

A source install does **not** pull in Suggests packages. Install these
manually: ``survey`` is required for the eBAC modules (without it
``morie pipeline --all`` stops short of 23/23), ``testthat`` for running
the R tests, and ``smotefamily`` (optional) for non-empty SMOTE outputs:

.. code-block:: r

   install.packages(c("survey", "testthat", "smotefamily"),
                    repos = "https://cloud.r-project.org/")

Other channels
--------------

.. code-block:: bash

   # One-line installer (Linux / macOS / WSL): detects pip and R, installs both
   curl -fsSL https://rootcoder007.github.io/morie/install.sh | bash

   # Homebrew (macOS / Linuxbrew)
   brew tap rootcoder007/morie
   brew install morie

   # Docker (zero local dependencies)
   docker run --rm ghcr.io/rootcoder007/morie:latest morie --help

macOS
-----

.. code-block:: bash

   brew install python r
   pip install morie

Linux
-----

.. code-block:: bash

   # Debian/Ubuntu
   sudo apt-get install r-base python3 python3-pip
   pip3 install morie

Windows
-------

.. code-block:: powershell

   winget install -e --id RProject.R
   winget install -e --id Python.Python.3.12
   pip install morie

The R package is checked on Windows in CI; native Windows and WSL2 are
both supported.

Verifying the install
---------------------

.. code-block:: bash

   morie list-modules          # prints the 23 registered analysis modules
   morie doctor                # checks LLM providers, datasets, R, Docker
   morie selftest              # smoke test of every subsystem
   morie --help

.. code-block:: r

   library(rmorie)
   morie_list_datasets()

LLM provider setup
------------------

The assistant (``morie ask``, ``morie chat``, ``morie percy``) tries
providers in priority order. No API key is needed for the default tier.

1. **Ollama** (local, private): install with
   ``curl -fsSL https://ollama.com/install.sh | sh``; the model is
   auto-detected from the running instance (``morie percysuits`` pulls the
   Perseus models).
2. **OllamaFreeAPI** (free community servers, no key): the client is
   vendored, nothing to set up.
3. **Gemini** (free tier): ``export GEMINI_API_KEY=...`` (free key at
   `aistudio.google.com <https://aistudio.google.com>`_); default model
   ``gemini-2.5-flash``.
4. **Local fallback**: automatic. Keyword-matched help text, no network.

Run ``morie doctor`` to see which providers are currently available.
