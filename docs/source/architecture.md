# MORIE architecture

*Part of {doc}`index` — high-level structural overview of the package.*

This page maps MORIE's public surface and the contracts between its
components. It complements the prose in the {doc}`methods/index`
reference and the {doc}`api/index`.

## The result-container spine

MORIE's public API is **function-based**: ~559 `morie_*` functions with
Python + R parity. Every result-emitting function returns a `RichResult`
— a `dict` subclass carrying a title, summary lines, tables, warnings, an
interpretation, and the raw payload — so any result prints as a readable
report and round-trips to JSON.

```{mermaid}
classDiagram
  class RichResult {
    +str title
    +list summary_lines
    +list tables
    +list warnings
    +str interpretation
    +Any payload
    +__str__()
    +to_json()
  }
```

A single design — `(data, treatment, outcome, covariates)` — flows
through any estimator function (ATE / ATT / AIPW / DML / matching /
Hawkes / …) and returns as a `RichResult`:

```{mermaid}
flowchart LR
  D["data:<br/>treatment · outcome · covariates"] --> F["morie_* estimator function"]
  F --> R["RichResult"]
```

## The data layer

The `DatasetRegistry` (`morie/data.py`) decouples loaders from analysis
code: a caller resolves a dataset *slug* to a `DataFrame` without knowing
which physical store it came from. The same slug can be served from the
bundled SQLite database shipped with the package, a local SQLite file, or
a remote SQL endpoint — selected by configuration, not by the caller.

```{mermaid}
flowchart LR
  S["slug"] --> REG["DatasetRegistry"]
  REG --> B["bundled SQLite"]
  REG --> L["local SQLite"]
  REG --> RM["remote SQL"]
  B --> DF["DataFrame"]
  L --> DF
  RM --> DF
```

## The MRM framework

The MRM (Multilevel Reconciliation Methodology) framework is a coordinated
set of `morie_*` functions — not a class hierarchy. Each MRM entry point
composes ~10 causal estimators on a single `(treatment, outcome,
covariates)` design (IPW Hájek, AIPW, g-computation, PSM 1:1 NN, PSM
5-strata, IRM-DML, PSM→IRM-DML, ATC AIPW, PLR-DML, SuperLearner-stacked
AIPW) and reports them in one aggregate `RichResult`, alongside a χ²
family and a Mandela (UN Rules 43/44) classifier.

```{mermaid}
flowchart LR
  D["one design:<br/>treatment · outcome · covariates"] --> MRM["MRM entry point"]
  MRM --> E["~10 estimator functions"]
  E --> A["aggregate RichResult"]
```
