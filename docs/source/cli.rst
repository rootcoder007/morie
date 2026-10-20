CLI Reference
=============

The ``morie`` command is the primary terminal interface. It is installed as a
console script entry point from ``morie.runner:main``. ``morie --help`` lists
every subcommand; ``morie <subcommand> --help`` lists its options. The
subcommands below are grouped by what they do. The R package ships the same
verbs as ``rmorie`` (see :doc:`learn/r-cli`), so everything on this page has
an R twin.

Analysis pipeline
-----------------

``list-modules``
   Print the 23 registered analysis modules with their output files.

``run-module <name>`` / ``run-modules --modules <a> <b>`` / ``pipeline``
   Run one, several, or all modules. Options shared by the three:

   ``--dataset KEY``
      A catalog dataset key (``morie list-datasets``), for example
      ``ocp21`` for the real CPADS PUMF. Overrides ``--cpads-csv``.
   ``--cpads-csv PATH``
      Path to a CPADS PUMF CSV. Without either option the modules use the
      real PUMF when it is on disk or already pulled into the cache
      (``morie pull ocp21`` once), and otherwise the bundled 1,200-row
      synthetic frame, with a warning that says so and names the command
      that gets the real one.
   ``--output-dir DIR``
      Directory for the CSV outputs.
   ``--all`` (``pipeline`` only)
      Run the whole module surface.
   ``-y`` / ``--yes`` (``pipeline`` only)
      Skip the confirmation prompt.
   ``--no-carbon`` (``pipeline`` only)
      Disable emissions tracking. By default a pipeline run writes
      ``emissions/emissions.csv`` under the output directory plus a
      provenance capsule for the run (:doc:`learn/emissions`) and prints
      the kg CO2eq at the end.

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
   correctness (exit status 1 when a check fails), or describe what a
   module-output CSV contains and how to read it. ``--module`` scopes
   ``inspect`` and ``verify`` to one module's expected outputs.

``generate-template``
   Write a methods-and-results scaffold for a first paper
   (``--module``, default ``power-design``; ``--out``, default
   ``first-paper.md``).

Datasets
--------

See :doc:`learn/datasets` for the walk-through.

``list-datasets``
   List the 71 catalogued dataset keys with type, cached row count and
   the **route** each one is obtained by: a portal it downloads from on
   first use (open.canada.ca, data.ontario.ca, Statistics Canada, CIHI,
   health-infobase.canada.ca, ECCC NAPS whose Canada-wide hourly keys take
   about ten minutes, Toronto Police ArcGIS), ``rmoriedata`` on CRAN,
   data.rmorie.com (the Health Infobase fallback copy and the OTIS research
   environments; the curated tables join the list once a key from rmorie.com/access is stored with ``morie login --token``),
   or "own file: data/..." for the one key that is your own research file
   (the MAPQ workbook), placed under ``$MORIE_DATA_DIR`` keeping that
   relative path.

``pull KEY [--out FILE.csv]`` / ``pull --all [--out DIR]``
   Download one dataset by catalog key (``ocp21``, ``cu23bt``, ...) or by
   shortcut (``tps-major``, ``tps-shootings``, ``tps-homicide``,
   ``tps-layers``, ``cpads``, ``siu-index``, and the ``-toy`` bundled
   frames) and write it as CSV (``--out``, else stdout); ``--all`` pulls
   every catalog key into a directory and reports the ones that fail.
   Downloads are cached in the dataset store, so the modules use the real
   data from then on. CKAN downloads fall back to the resource file and
   then to its Internet Archive (Wayback Machine) snapshot when the live
   portal is down. ``--year`` and ``--max`` apply to the TPS feeds.

``download-bootstrap``
   Fetch the large bootstrap-weight files from the Statistics Canada CKAN
   API (``--survey csads_2021|csads_2023|csus_2019|csus_2023|all``,
   ``--limit``).

``ingest {ckan,tps,siu,a2aj}``
   Pull open-data feeds: CKAN portals, Toronto Police Service ArcGIS
   layers, SIU director's-report PDFs, and A2AJ Canadian Legal Data
   (``a2aj coverage``, ``a2aj search QUERY``, ``a2aj fetch CITATION``).

``profile-dataset PATH`` (or ``--csv PATH``)
   Infer measurement levels and variable roles for any tabular file
   (CSV, TSV, Excel, Parquet). ``--treatment``, ``--outcome`` and
   ``--weights`` are hints; ``--suggest`` prints an analysis plan.

``sample PATH --n N --method {srs,stratified,cluster,pps}``
   Draw a sample (``--strata-col``, ``--cluster-col``, ``--size-col``,
   ``--proportional``, ``--seed``, ``--output``).

Assistant and LLM
-----------------

Every assistant verb answers through the model you brought (:doc:`hosted`): your own
OpenAI-compatible server, then each provider whose key is set (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, ...), then an Ollama server with a model, then a local keyword fallback
that says it is one. Nothing is hosted for you.

``ask QUESTION`` / ``agent QUESTION`` / ``percy QUESTION`` (alias ``perseus``)
   Ask the MORIE assistant (streams by default; ``--no-stream``,
   ``--context``, ``--model NAME``). ``percy`` and ``agent`` start the
   local tool-calling agent when Ollama is present and otherwise answer
   through the chain instead of printing a connection error.

``login [--no-browser] [--email ADDRESS [--to-email]] [--token [KEY]]``
   Store or mint the data key for the curated tables at data.rmorie.com
   (:doc:`hosted`; no model uses it). With no options: the GitHub device flow (``--no-browser`` prints the URL instead of opening
   it). ``--email`` asks for a one-time code sent to that address;
   ``--to-email`` additionally has the key itself emailed to you instead
   of stored on this machine. ``--token`` stores a key you already have
   (prompts for it when ``KEY`` is omitted) and checks it with the data
   service before it is kept. A key the service rejects (401) is reported as
   "run ``morie login`` again", not as the service being unreachable.

``models [--route ROUTE]``
   List the models the route ``ask`` takes offers (or another route's), the
   one in use marked ``*``. Pick one per call with ``morie ask --model NAME``
   or save it with ``morie config set PROVIDER.model NAME``.

``provider set --base-url URL --key KEY [--model NAME]`` / ``provider show`` / ``provider unset``
   Attach your own OpenAI-compatible endpoint: OpenAI, Anthropic's
   compatibility endpoint (``https://api.anthropic.com/v1``), OpenRouter,
   Mistral, Groq, a local LM Studio / vLLM / llama.cpp server, anything
   that serves ``/chat/completions``. The setting lives in the credentials
   file both languages read; ``LLM_API_BASE_URL``, ``LLM_API_KEY`` and
   ``MORIE_API_MODEL`` in the environment take precedence.

``logout``
   Forget the stored data key.

``chat``
   Interactive chat REPL with slash commands (``--agent`` loads a persona).

``serve``
   Start a Perseus relay server (``--port``, default 8421; ``--bind``,
   default 127.0.0.1; ``--token``).

``percysuits``
   Pull all Perseus LLM models into Ollama (``--host``, ``--dry-run``,
   ``--ssh user@host``).

Compute emissions and provenance
--------------------------------

``emissions [--seconds N] [--output-dir DIR] [--country ISO3] [--no-capsule]``
   Load the CPU for ``N`` seconds (default 3) under the tracker and print
   the energy, the kg CO2eq, the grid intensity used and where the
   capsule went. The same tracker runs under ``pipeline``. Method, columns
   and the capsule files: :doc:`learn/emissions` and
   :doc:`methods/compute-emissions`.

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

``interactive install|status|remove``
   Add the REPL/exec/agent/TUI modules that the published package leaves out,
   verified against the bundled manifest, for the current user.

``tutorial`` / ``cheatsheet``
   Interactive first-time walkthrough; one-page command reference.

Environment
-----------

``doctor``
   Diagnostics for the language-model routes (your own endpoint, provider
   keys, Ollama, the route ``ask`` takes), the data key, datasets, R and
   Docker; ``--fix`` tries to remediate.

``selftest``
   Quick smoke test of every subsystem (also the Docker health check).

``update``
   Check PyPI for a newer release and optionally install it (``-y``).

``bricklayer``
   Offer to install the rest of the morie family (the R packages) and
   verify the shared C/C++ backend (``--check`` reports only).

``r-install``
   Install the R side: ``rmorie`` from r-universe (prebuilt, pulls
   ``rmoriedata`` and ``rmoriebricklayer``), or with ``--github`` this
   repository's own R arm, ``r-package/morie``, built from source with
   remotes. Same options as ``bricklayer``.

Data capsules
-------------

``capsule init CSV --url URL``
   Pin a CSV: write ``data_provenance.json`` beside it (name, source URL and
   retrieval date, the file's SHA-256, size and data rows; ``--script FILE``
   pins an analysis script too).

``capsule pin DIR``
   Re-pin a capsule's provenance to its files as they are now.

``capsule verify DIR``
   The checks of ``rmoriebricklayer::vrfcps()``, and of ``cpsbndlvrf()`` on
   ``capsule_bundle.json`` when present (``--key-env VAR`` or
   ``--key-file FILE`` supplies the HMAC key; ``--public-key HEX`` pins the
   key: an hmac key id, or the ML-DSA or XMSS public key ``capsule keygen``
   printed). An ML-DSA or XMSS signature is verified under the public key
   the bundle carries, so it needs no key.

``capsule keygen --scheme SCHEME --out FILE``
   A new ``ML-DSA-44``, ``ML-DSA-65``, ``ML-DSA-87`` (FIPS 204) or
   ``xmss-sha256`` (RFC 8391; ``--height H``, 1-16, signs ``2^H`` bundles,
   default 10) signing key, written as JSON readable by its owner only (the
   fields of ``fipskg()`` / ``pqckg()``; never overwritten). Prints the
   public key; ``--public-out FILE`` also writes it as JSON.

``capsule sign DIR [--scheme SCHEME]``
   Write ``capsule_bundle.json``: every file's SHA-256, the digest of
   ``manifest.json``, and an attestation over them, as ``cpsbndl()`` writes
   it. ``--scheme hmac`` (the default) signs with the shared secret in
   ``$MORIE_CAPSULE_KEY`` (or ``--key-env`` / ``--key-file``; never on the
   command line); rmoriebricklayer >= 1.0.0 (srr) verifies it with the
   shared key (``cpsbndlvrf(key = ...)``). ``--scheme ML-DSA-44|65|87`` and
   ``--scheme xmss-sha256`` sign with a key from ``capsule keygen`` (or an R
   key written with ``tojs()``), from the same places; ``--context`` and
   ``--prehash`` apply to ML-DSA. ``cpsbndlvrf()`` verifies these bundles
   with no key, and rmoriebricklayer's bundles of these schemes verify
   here. An XMSS key must be a ``--key-file``: its ``next_index`` is
   written back after each signature, and the used-index record morie
   shares with rmoriebricklayer (``RMBL_XMSS_INDEX_FILE``, else
   ``xmss_used_indices.txt`` in ``tools::R_user_dir("rmoriebricklayer",
   "config")``) refuses an index already used.

``capsule drift REF CUR``
   Compare a capsule's pinned CSV (or any CSV) with a current one, column by
   column: Kolmogorov-Smirnov or chi-square, population stability index,
   Jensen-Shannon divergence, added and removed columns, and Benford's law
   on the numeric columns.

Every ``capsule`` command takes ``--json``; the exit status is 0 when
everything verified (no drift), 1 when a check failed or drift was found,
and 2 on a usage or input error, as in rmbl-action.

Verification pipelines
----------------------

``verify-pollution --pollutant {no2,pm25}``
   Run the pollution-to-health pipeline of :doc:`methods/envhealth`
   (concentration-response, attributable fraction, deaths displaced,
   burden, equity index) and print a report with an assumption log; exit
   status 1 when an assumption fails, 2 on a data error. Exposure comes
   from ``--demo`` (synthetic), ``--exposure-csv FILE`` (an ``exposure``
   column, optional ``income``), or ``--exposure-mean X
   --exposure-prevalence P``; ``--outcome``, ``--reference``,
   ``--baseline-rate`` (per 100,000 per year), ``--population``,
   ``--region``, ``--years``, ``--json``.

``verify-earth-engine``
   Smoke-check Earth Engine credentials, initialisation and a one-pixel
   query (``--skip-query``, ``--json``). Python only: the Earth Engine
   client has no R counterpart, and ``rmorie verify-earth-engine`` says so.

Models and cryptography
-----------------------

``convert-checkpoint --checkpoint PT --output GGUF``
   Convert a ``.pt`` training checkpoint to GGUF, optionally
   TurboQuant-compressed (``--turbo-bits 2|3|4``, ``--tokenizer-dir``).

``crypto {keygen,encrypt,decrypt}``
   Post-quantum file encryption (ML-KEM-768 + ChaCha20-Poly1305):
   ``keygen [--name NAME] [--output DIR]`` (keystore at
   ``~/.morie/keys/keystore.json`` unless ``--output``), ``encrypt FILE
   --to PKFILE|NAME``, ``decrypt FILE --key NAME``.

Repo-root wrapper
-----------------

A shell wrapper ``./morie`` at the repository root invokes
``.venv/bin/python -m morie.runner`` so the venv does not need to be activated
manually:

.. code-block:: bash

   ./morie list-modules
   ./morie pipeline --all -y
