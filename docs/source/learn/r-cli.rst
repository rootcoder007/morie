The R command line: rmorie
==========================

The R package ships the same command line as the Python package. After

.. code-block:: r

   install.packages("rmorie", repos = c("https://rootcoder007.r-universe.dev", "https://cloud.r-project.org"))
   rmorie::install_cli()        # links `rmorie` into ~/.local/bin

every verb below works from a shell, and every one is also an exported R
function (``rmorie::morie_cli("selftest")`` runs a verb from R).

.. code-block:: bash

   rmorie help                  # the full list; VERB --help for one
   rmorie cheatsheet            # one page

Verbs, grouped as in :doc:`../cli`:

- analysis: ``list-modules``, ``run-module``, ``run-modules``, ``pipeline``
  (with emissions tracking and a capsule), ``inspect``, ``verify``,
  ``explain``, ``generate-template``, ``selftest``, ``tutorial``;
- data: ``list-datasets``, ``pull KEY`` / ``pull --all``,
  ``download-bootstrap``, ``ingest ckan|tps|siu|a2aj``, ``profile-dataset``,
  ``sample``;
- assistant: ``login``, ``logout``, ``models``, ``doctor``, ``ask``,
  ``percy`` / ``agent``, ``chat``, ``provider set|show|unset``,
  ``percysuits``;
- emissions and pollution: ``emissions``, ``verify-pollution``;
- utilities: ``crypto keygen|encrypt|decrypt``, ``exec``, ``edit``,
  ``update``, ``version``.

What differs from Python: ``exec`` evaluates R code; ``tui``, ``repl``,
``serve``, ``convert-checkpoint`` and ``r-install`` have no R twin;
``verify-earth-engine`` needs the Earth Engine Python client and says so.
The CPADS resolution order, the credentials file, the emissions CSV layout
and the capsule format are shared, so the two packages can be mixed in one
workflow.
