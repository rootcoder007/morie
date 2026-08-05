# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Bayes factor between two models -- the same method as bayfac.

This stub and ``morie.fn.bayfac`` declare the same public function name
(``bayes_factor``), the same formula (the ratio of marginal likelihoods)
and the same citation (Kass and Raftery 1995).  They are one method
under two module names; only the argument labels differ, ``log_lik_a``
and ``log_lik_b`` here against ``log_evidence_1`` and
``log_evidence_2`` there.

Those labels are worth a word, because they are not interchangeable in
general.  A Bayes factor is a ratio of *marginal* likelihoods -- the
likelihood integrated over the prior -- not of maximised likelihoods.
Feeding in two maximised log-likelihoods gives a likelihood ratio, which
is a different quantity and one that always favours the larger model.
The argument is treated here as what the formula requires, a log
marginal likelihood, whatever the stub happened to name it.

There is exactly one implementation; this module delegates.  Recorded in
ledger/wave2/DUPMAP.tsv as bfac -> bayfac.

Distinct from ``bayesf`` and ``bbf`` (BIC approximations to the Bayes
factor) and from ``bfact`` (Savage-Dickey density ratio), which are
different estimators of the same target and must not be collapsed.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from .bayfac import bayes_factor as _bf

__all__ = ["bayes_factor"]


def bayes_factor(log_lik_a, log_lik_b):
    """B_ab from two log marginal likelihoods; see morie.fn.bayfac."""
    return _bf(log_lik_a, log_lik_b)


def cheatsheet():
    return "bfac: Bayes factor (shares bayfac's implementation)"


# compact alias per ledger/NAMING.md
bayesfactor = bayes_factor
