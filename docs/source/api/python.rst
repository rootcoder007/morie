Python API
==========

Part of :doc:`index` — MORIE API reference.

Reference for every public ``morie.*`` module.
Signatures and docstrings come from ``sphinx.ext.autodoc``; see
:doc:`../methods/index` for the methodology behind each function.

Causal inference
----------------

.. automodule:: morie.causal
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.effects
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.matching
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.iv
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.rdd
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.did
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.sensitivity
   :members:
   :undoc-members:
   :show-inheritance:

Survey + descriptive statistics
-------------------------------

.. automodule:: morie.survey
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.sampling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.subpop_design
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.survival
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.statistics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.bootstrap_methods
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.multiple_testing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.effect_sizes
   :members:
   :undoc-members:
   :show-inheritance:

Datasets
--------

.. automodule:: morie.dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.bq
   :members:
   :undoc-members:
   :show-inheritance:

OTIS — Offender Tracking Information System
-------------------------------------------

.. automodule:: morie.otis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.otis_analyze
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.otis_all_analyze
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.otis_causal
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.otis_churn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.otis_datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.otis_tps_overlay
   :members:
   :undoc-members:
   :show-inheritance:

TPS — Toronto Police Service
----------------------------

.. automodule:: morie.tps_io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_crime
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_csi
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_temporal
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_spatial
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_spatial_advanced
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_stochastic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_hawkes_advanced
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_statphysics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_render
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_all_analyze
   :members:
   :undoc-members:
   :show-inheritance:

Federal SIU — Sprott / Doob / Iftene replication
------------------------------------------------

.. automodule:: morie.sprott_doob
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.siuiap
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.doob_trends
   :members:
   :undoc-members:
   :show-inheritance:

Psychometrics
-------------

.. automodule:: morie.psymet
   :members:
   :undoc-members:
   :show-inheritance:

Entheogenic neuroimaging — DMT EEG-fMRI
----------------------------------------

Opt-in module wrapping the Carhart-Harris / Timmermann DMT-imaging
dataset (20 subjects EEG + parcellated fMRI; the 15 motion-survived
subjects are 01-03 and 06-17). Exposes two consciousness-theory
metrics: Beautiful Loop (Bayne, Carter, Laukkonen, Slagter) and
Self-Aware Networks (Pirez). Data location is honoured via
``$MORIE_DMT_IMAGING_ROOT``; a deterministic synthetic fixture is
returned when the local mirror is absent so CI keeps running.

.. automodule:: morie.entheo
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.entheo.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.entheo.preprocess
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.entheo.analysis
   :members:
   :undoc-members:
   :show-inheritance:

The MRM framework
-----------------

Multilevel Reconciliation Methodology entry points and their primitives.

.. automodule:: morie.mrm_flagship
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_design
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_doe
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_primitives
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_mathstats
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_graphs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_otis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_siu
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_tps
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_uof
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_stockflow
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_mandela_spectrum
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_kulldorff
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.mrm_lisa
   :members:
   :undoc-members:
   :show-inheritance:

Federal SIU, crime feeds and forensics
--------------------------------------

.. automodule:: morie.siu
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.siu_fetch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_fetch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tps_hawkes_jit
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.hawkes_spatial
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.cpd_all_analyze
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.nypd_all_analyze
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.run_crime_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.arsau_analyze
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.arsau_datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.laniyonu
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.taphonomy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.investigation
   :members:
   :undoc-members:
   :show-inheritance:

Datasets, ingestion and data hygiene
------------------------------------

.. automodule:: morie.datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.datasets_vic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.dataset_dictionary
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.ingest
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.ingest.a2aj
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.ingest.canlii
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.schema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.variable_taxonomy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.audit_variables
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.categorical_guard
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.validation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.bricklayer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.cpads
   :members:
   :undoc-members:
   :show-inheritance:

Further estimators and diagnostics
----------------------------------

.. automodule:: morie.dml_clustered
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.ebac
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.fairness
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.missing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.meta_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.ml
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.weights
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.inference
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.signal
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.longitudinal_sim
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.stat_bridge
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.semipar_bridge
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.envhealth
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.earth
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.entheo_dmt
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tox
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.crypto
   :members:
   :undoc-members:
   :show-inheritance:

Reporting and output
--------------------

.. automodule:: morie.reporting
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tables_pub
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.export
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.explain
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.viz
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.animate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.notebook
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.eval
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.eval_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Assistant, LLM and model tooling
--------------------------------

.. automodule:: morie.agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.chat
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.llm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.perseus
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.perseus_relay
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.vertex
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.engine
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.engine_bridge
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.quant
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.quant_bridge
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.kv_cache
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.gguf_loader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.pt2gguf
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.polyglot
   :members:
   :undoc-members:
   :show-inheritance:

Runner and environment
----------------------

.. automodule:: morie.runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.modules
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.doctor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.selftest
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.emissions
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.container
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.cheatsheet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.tutorial
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: morie.stat_commands
   :members:
   :undoc-members:
   :show-inheritance:

Function namespace ``morie.fn``
---------------------------------

The ``morie.fn`` namespace exposes 18,560 individual callables,
indexed by a registry and resolved lazily on first access. To keep the
wheel small, the implementations and per-callable guides ship as two
compressed archives (``_fnsrc.json.xz`` and ``describe_docs.json.xz``,
~7 MB total) rather than tens of thousands of loose files; the importer resolves
``morie.fn.<name>`` from them transparently, and per-callable
documentation is available at runtime via ``morie.fn.describe``. The
full registry is the canonical catalogue:

.. automodule:: morie.fn._registry
   :members:
   :undoc-members:

Result containers shared across the package:

.. automodule:: morie.fn._containers
   :members:
   :undoc-members:

.. automodule:: morie.fn._richresult
   :members:
   :undoc-members:
