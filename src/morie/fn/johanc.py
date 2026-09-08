# morie.fn -- slice k04 (rootcoder007/morie)
"""Johansen cointegration test -- re-export of :mod:`morie.fn.johsn`.

This module used to carry a verbatim one-sample Kolmogorov-Smirnov test
against a fitted normal, pasted here by the stub generator.  That has
nothing to do with cointegration and is deleted.

A correct Johansen (1988, 1991) reduced-rank test already lives in
``morie.fn.johsn``; duplicating it here would be a second place to keep
in sync, so this module re-exports it.  Reference implementation read
while checking ``johsn``: ``urca::ca.jo`` (Pfaff, urca 1.3-4,
``R/ca-jo.R``), which forms S00/S0K/SK0/SKK from the partialled-out
VECM regressors and takes the eigenvalues of
``C^-1 SK0 S00^-1 S0K C^-T`` with ``C C^T = SKK``; the trace statistic
is ``-T sum_{i>r} log(1 - lambda_i)``.
"""

from __future__ import annotations

from .johsn import johansen_cointegration

__all__ = ["johansen_cointegration"]


def cheatsheet():
    return "johanc: Johansen cointegration test (see morie.fn.johsn)"
