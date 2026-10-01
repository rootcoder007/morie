Get the real data
=================

Nothing in MORIE depends on a dataset being on your machine already. The
packages ship the catalogue (70 keys), the provenance records and a few small
synthetic frames; the real files are downloaded on first use from the
portals that publish them and cached, so every later call is local.

See what exists
---------------

.. code-block:: bash

   morie list-datasets          # key, name, source portal, survey, year, cached rows
   rmorie list-datasets         # the same table from the R package

.. code-block:: r

   rmorie::morie_list_datasets()

Each row says where the data comes from: a portal it is pulled from
(open.canada.ca, data.ontario.ca, Statistics Canada, CIHI, ECCC, the Toronto
Police ArcGIS hub), ``rmoriedata`` on CRAN (sample frames and the provenance
records), or "own file" for restricted data you drop under
``$MORIE_DATA_DIR`` yourself.

Pull one dataset, or all of them
--------------------------------

.. code-block:: bash

   morie pull ocp21 --out cpads.csv        # the real CPADS 2021-2022 PUMF, ~41k rows
   morie pull cu23bt --out csus-boot.csv   # a CSUS bootstrap-weight file
   morie pull --all --out datasets/        # every catalog key; failures listed, the rest written
   rmorie pull ocp21 --out cpads.csv       # identical from R
   rmorie pull --all --out datasets/

.. code-block:: r

   cpads <- rmorie::morie_load_dataset("ocp21")

.. code-block:: python

   from morie.data import load_dataset
   cpads = load_dataset("ocp21")

The download lands in the dataset store (``~/.cache/morie`` in Python, the
package's SQLite store in R), so the second call does not touch the
network. When a CKAN portal is down the packages fall back to the resource
file itself and then to its Internet Archive (Wayback Machine) snapshot,
and say which copy they used.

What the modules use
--------------------

``morie run-module NAME`` (and ``pipeline``, and their ``rmorie`` twins)
pick the CPADS frame in this order:

1. the real PUMF checked out at
   ``data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv``;
2. the real PUMF already pulled into the store (``morie pull ocp21`` /
   ``rmorie pull ocp21``, once);
3. the 1,200-row synthetic frame, with a message that says so and names
   the command that gets the real one.

To be explicit, pass ``--dataset KEY`` (any catalog key) or
``--cpads-csv PATH`` / ``--cpads FILE``.

.. code-block:: bash

   morie run-module power-design --dataset ocp21 --output-dir out/
   rmorie run-module power-design --dataset ocp21 --output-dir out/

Curated tables at data.rmorie.com
---------------------------------

Beyond the open portals, the project keeps 160 curated databases built from
Google BigQuery public datasets (Chicago crime, EPA air quality, US census,
FEC, FDA, NOAA, NHTSA, Hacker News, Ethereum, World Bank, ...) and serves
their tables from the edge at https://data.rmorie.com. They open with the
same key ``morie login`` stores for the hosted model tier, and they do not
depend on any project machine being up.

.. code-block:: bash

   morie login                                   # once
   morie list-datasets                           # the curated tables appear with route "data.rmorie.com"
   morie pull chicago_crime/incidents --out incidents.csv
   rmorie pull epa_pm25_daily/epa_pm25_daily --out pm25.csv

.. code-block:: python

   from morie.data import load_dataset
   df = load_dataset("chicago_crime/incidents")       # cached in the dataset store afterwards

.. code-block:: r

   df <- rmorie::morie_load_dataset("chicago_crime/incidents")

``https://data.rmorie.com/browse`` opens any of the databases in the browser
(tables, rows, SQL) with no server behind it. Keys are ``db/table``. ``https://data.rmorie.com/manifest.json`` (with the key
as ``Authorization: Bearer``) lists every table with rows, columns, size,
SHA-256 and the BigQuery source it was materialised from; each source
dataset carries its own licence, named there. ``MORIE_DATA_URL`` points the
packages at another gateway.

Other feeds
-----------

``morie ingest ckan|tps|siu|a2aj ...`` and ``rmorie ingest ...`` pull open
portals directly (CKAN package search and download, Toronto Police ArcGIS
layers with a year filter, SIU director's reports, A2AJ Canadian legal
data); ``download-bootstrap`` caches the Statistics Canada bootstrap-weight
files. The options are listed in :doc:`../cli`.
