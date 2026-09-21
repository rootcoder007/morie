CLI Reference
=============

The ``morie`` command is the primary terminal interface. It is installed as a
console script entry point from ``morie.runner:main``. ``morie --help`` lists
every subcommand; ``morie <subcommand> --help`` lists its options. The
subcommands below are grouped by what they do.

Analysis pipeline
-----------------

``list-modules``
   Print the 23 registered analysis modules with their output files.

``run-module <name>`` / ``run-modules --modules <a> <b>`` / ``pipeline``
   Run one, several, or all modules. Options shared by the three:

   ``--dataset KEY``
      Built-in dataset key (``morie list-datasets``), for example ``ocp21``.
      Overrides ``--cpads-csv``.
   ``--cpads-csv PATH``
      Path to a CPADS PUMF CSV. Without either option the modules run on the
      bundled 1,200-row synthetic CPADS frame and say so.
   ``--output-dir DIR``
      Directory for the CSV outputs.
   ``--all`` (``pipeline`` only)
      Run the whole module surface.
   ``-y`` / ``--yes`` (``pipeline`` only)
      Skip the confirmation prompt.
   ``--no-carbon`` (``pipeline`` only)
      Disable emissions tracking for the run.

   .. code-block:: bash

      morie run-module power-design --output-dir /tmp/morie-outputs
      morie pipeline --all -y --dataset ocp21

   .. note::

      The full 23/23 run needs the R package installed plus the ``survey``
      R package (required by the eBAC modules; it is in Suggests, so a
      source install does not pull it in): ``install.packages("survey")``.
      ``smotefamily`` is optional; without it the SMOTE outputs are
      status-only.

``inspect PATH`` / ``verify PATH`` / ``explain FILENAME``
   Browse output CSVs (schema, rows, summary statistics), validate them for
   correctness, or describe what a module-output CSV contains. ``--module``
   scopes ``inspect`` and ``verify`` to one module's expected outputs.

``generate-template``
   Write a methods-and-results scaffold for a first paper
   (``--module``, default ``power-design``; ``--out``, default
   ``first-paper.md``).

Datasets
--------

``list-datasets``
   List the 69 built-in dataset keys with type, row count and cache status.

``pull KEY``
   One-line dataset shortcut (the CLI for ``morie.datasets``); writes a CSV
   (``--out``, else stdout). ``--year`` and ``--max`` apply to the TPS feeds.

``download-bootstrap``
   Fetch the large bootstrap-weight files from the Statistics Canada CKAN
   API (``--survey csads_2021|csads_2023|csus_2019|csus_2023|all``,
   ``--limit``).

``ingest {ckan,tps,siu,a2aj}``
   Pull open-data feeds: CKAN portals, Toronto Police Service ArcGIS
   layers, SIU director's-report PDFs, and A2AJ Canadian Legal Data
   (``a2aj coverage``, ``a2aj search QUERY``, ``a2aj fetch CITATION``).

``profile-dataset --csv PATH``
   Infer measurement levels and variable roles for any tabular file
   (CSV, TSV, Excel, Parquet). ``--treatment``, ``--outcome`` and
   ``--weights`` are hints; ``--suggest`` prints an analysis plan.

``sample --csv PATH --method {srs,stratified,cluster,pps}``
   Draw a sample (``--n``, ``--strata-col``, ``--cluster-col``,
   ``--size-col``, ``--proportional``, ``--seed``, ``--output``).

Assistant and LLM
-----------------

``ask QUESTION`` / ``agent QUESTION``
   Ask the MORIE assistant (streams by default; ``--no-stream``,
   ``--context``, ``--model``). No API key is needed: providers are tried
   in the order listed under :doc:`install`.

``chat``
   Interactive chat REPL with slash commands (``--agent`` loads a persona).

``percy`` (alias ``perseus``)
   Talk to Perseus, the MORIE expert agent. ``--local`` forces Ollama only,
   ``--freeapi`` the free community servers, ``--remote`` or
   ``--cloud URL`` a Perseus relay, ``--pi HOST`` a Raspberry Pi.

``serve``
   Start a Perseus relay server (``--port``, default 8421; ``--bind``,
   default 127.0.0.1; ``--token``).

``percysuits``
   Pull all Perseus LLM models into Ollama (``--host``, ``--dry-run``,
   ``--ssh user@host``).

Editors and REPLs
-----------------

``tui``
   Full-screen terminal IDE (needs the ``interactive`` extra).

``edit FILE``
   Built-in scientific editor (``--run`` executes on save, ``--lang``).

``repl``
   Headless polyglot REPL with cross-language variable bridging
   (``--lang``, ``--no-polyglot``, ``--no-detect``).

``exec``
   Execute code inline, from stdin, or from ``--file``; ``--lang python|r``.

``tutorial`` / ``cheatsheet``
   Interactive first-time walkthrough; one-page command reference.

Environment
-----------

``doctor``
   Diagnostics for LLM providers, datasets, R and Docker; ``--fix`` tries to
   remediate.

``selftest``
   Quick smoke test of every subsystem (also the Docker health check).

``update``
   Check PyPI for a newer release and optionally install it (``-y``).

``bricklayer``
   Offer to install the rest of the morie family (the R packages) and
   verify the shared C/C++ backend (``--check`` reports only).

Verification pipelines
----------------------

``verify-pollution --pollutant {no2,pm25}``
   Run a pollution-to-health causal pipeline and print a report
   (``--demo`` uses synthetic data; ``--json`` for machine output).

``verify-earth-engine``
   Smoke-check Earth Engine credentials, initialisation and a one-pixel
   query (``--skip-query``, ``--json``).

Models and cryptography
-----------------------

``convert-checkpoint --checkpoint PT --output GGUF``
   Convert a ``.pt`` training checkpoint to GGUF, optionally
   TurboQuant-compressed (``--turbo-bits 2|3|4``, ``--tokenizer-dir``).

``crypto {keygen,encrypt,decrypt}``
   Post-quantum utilities (ML-KEM-768 + ChaCha20).

Repo-root wrapper
-----------------

A shell wrapper ``./morie`` at the repository root invokes
``.venv/bin/python -m morie.runner`` so the venv does not need to be activated
manually:

.. code-block:: bash

   ./morie list-modules
   ./morie pipeline --all -y
