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

   pip install morie                  # the package, 71 catalogued datasets, 15,222 morie.fn callables
   pip install "morie[interactive]"   # + the Terminal IDE (textual)
   morie interactive install          # the REPL, exec, agent and TUI modules, once per user

``morie interactive install`` fetches the five modules behind ``morie repl``,
``morie exec``, ``morie agent`` and ``morie tui`` from the release tag of the
version you installed and checks each against the manifest inside the package;
every published artifact leaves them out because they run code you or a model
type. Without them those verbs print the command and, on a terminal, offer to
run it. ``morie interactive status`` and ``morie interactive remove`` manage the
per-user copy; a source checkout needs none of this.

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
corpus). All three are served from r-universe; CRAN carries older
companions, so keep r-universe first and name them, which replaces an older
copy already installed:

.. code-block:: r

   if (!requireNamespace("pak", quietly = TRUE)) {
     install.packages("pak", repos = "https://cloud.r-project.org")
   }
   pak::repo_add(rootcoder007 = "https://rootcoder007.r-universe.dev")
   pak::pkg_install(c("rmoriebricklayer", "rmoriedata", "rmorie"))
   library(rmorie)

From a terminal, or without pak (keep ``repos``: ``Rscript`` has no mirror
chooser):

.. code-block:: sh

   Rscript -e 'if (!requireNamespace("pak", quietly = TRUE)) install.packages("pak", repos = "https://cloud.r-project.org"); pak::repo_add(rootcoder007 = "https://rootcoder007.r-universe.dev"); pak::pkg_install(c("rmoriebricklayer", "rmoriedata", "rmorie"))'
   Rscript -e 'install.packages(c("rmoriebricklayer", "rmoriedata", "rmorie"), repos = c("https://rootcoder007.r-universe.dev", "https://cloud.r-project.org"))'

On macOS, CRAN's R installs r-universe's prebuilt binaries; Homebrew's R
cannot use them and compiles every package from source.

The copy under ``r-package/morie`` in the repository is the same code under
the package name ``morie``; it is what the R API pages on this site are
built from, and it installs from source with

.. code-block:: r

   install.packages(c("rmoriebricklayer", "rmoriedata"),
                    repos = c("https://rootcoder007.r-universe.dev",
                              "https://cloud.r-project.org"))
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

Debian, Ubuntu and Raspberry Pi OS mark the system Python "externally
managed" (PEP 668), so a bare ``pip install`` stops with an error. Install
into a virtual environment; it needs only the stock ``python3``:

.. code-block:: bash

   # Debian/Ubuntu (add python3-venv if `venv` is missing)
   sudo apt-get install -y r-base python3 python3-venv
   python3 -m venv ~/.venvs/morie
   source ~/.venvs/morie/bin/activate
   pip install -U morie
   morie --version

Activate the venv in each new shell (``source ~/.venvs/morie/bin/activate``)
or call ``~/.venvs/morie/bin/morie`` directly. Other distributions: ``pip
install morie`` in any Python 3.10+ environment.

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
providers in this order and uses the first one that answers. Nothing
needs configuring for the first and last tiers.

1. **Ollama** (local, private): install with
   ``curl -fsSL https://ollama.com/install.sh | sh``; the model is
   auto-detected from the running instance (``morie percysuits`` pulls the
   Perseus models). Override with ``MORIE_OLLAMA_MODEL``.
2. **Gemini**: ``export GEMINI_API_KEY=...`` (key at
   `aistudio.google.com <https://aistudio.google.com>`_); default model
   ``gemini-2.5-flash`` (``GEMINI_MODEL`` overrides).
3. **Any OpenAI-compatible endpoint**: ``LLM_API_BASE_URL`` plus
   ``LLM_API_KEY``.
4. **OpenAI**: ``export OPENAI_API_KEY=...``; default model ``gpt-4o-mini``.
5. **Hosted MORIE tier** (``https://llm.rmorie.com``), the last resort when
   nothing above answers: request a key at https://rmorie.com/access and
   store it with ``morie login --token`` (``morie login``, the GitHub
   sign-in, and ``morie login --email you@example.com`` remain for accounts
   that have them). The key is stored owner-only in
   ``$XDG_CONFIG_HOME/morie/credentials.json`` (``~/.config/morie/`` by
   default) and shared with the R package. Rate-limited per key; no
   prompts or responses are stored. Details in :doc:`hosted`.
6. **Local fallback**: automatic. Keyword-matched help text, no network.

Run ``morie doctor`` to see which providers are currently available.
