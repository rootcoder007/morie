# morie.fn -- wave3 slice w5_08 (rootcoder007/morie)
"""Robust event-study coefficients -- alias of :mod:`morie.fn.boryis`.

Borusyak, K., Jaravel, X. and Spiess, J. (2024), "Revisiting Event-Study
Designs: Robust and Efficient Estimation", Review of Economic Studies
91(6):3253-3285, doi:10.1093/restud/rdae007. Source used:
arXiv:1806.01221, local copy
/run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
borusyak-jaravel-spiess-2024-revisiting-event-study-designs.pdf.
Implemented: the imputation estimator of their Section 3 -- fit unit and
period effects on untreated observations only, impute Y(0) for treated
cells, average tau_it = Y_it - Yhat_it(0) within each relative time to
get the event-study path (the ``event`` and ``pretrend_by_rel`` outputs).

PROVENANCE NOTE (docstring-is-not-the-spec): the stub blurb said
"event-study coefficients with relative-time dummies", which describes
the dynamic TWFE specification that Borusyak-Jaravel-Spiess (and Sun and
Abraham 2021) show is CONTAMINATED under heterogeneous effects --
that naive specification already exists as
:mod:`morie.fn.evstud` (``event_study_coefficients``), with the caveat
documented there. The cited paper's own estimator is the imputation
estimator, which is what this alias binds to; it is implemented in
:mod:`morie.fn.boryis` as ``borusyak_jaravel_spiess``.

R arms in both trees: ``morie_did_borusyak``
(r-package/morie/R/did_modern_native.R and r-morie-oss mirror), anchored
against morie.fn on the staggered fixture in test-did-parity.R
(overall 2.05, dynamic path 1 + 0.5 r, exact by design).
"""

from __future__ import annotations

from .boryis import (  # noqa: F401
    borusyak_jaravel_spiess,
    impute_untreated,
)

__all__ = ["causdidev", "causal_did_eventstudy", "borusyak_jaravel_spiess",
           "impute_untreated"]

# primary name = module name; stub-era long name kept as alias.
causdidev = borusyak_jaravel_spiess
causal_did_eventstudy = borusyak_jaravel_spiess


def cheatsheet():
    return ("causdidev: Borusyak-Jaravel-Spiess (2024) imputation "
            "event study -- alias of boryis.borusyak_jaravel_spiess; "
            "for the naive relative-time-dummy TWFE see evstud")
