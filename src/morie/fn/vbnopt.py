# morie.fn -- function file (rootcoder007/morie)
"""Variational inference by ELBO maximisation -- alias of :mod:`morie.fn.vinfer`.

DUPLICATE, resolved by aliasing (wave-2 DUPMAP pairs vbnopt and vinfer
with each other; one is implemented, the other re-exports it).  Both
name the same procedure from the same paper: Jordan, M.I., Ghahramani,
Z., Jaakkola, T.S. and Saul, L.K. (1999), "An Introduction to
Variational Methods for Graphical Models", *Machine Learning*
37(2):183-233, doi:10.1023/A:1007665907178 -- maximise

    ELBO(q) = E_q[log p(x, z)] - E_q[log q(z)]

over a factorised q by the coordinate update
log q*_j = E_{q_{-j}}[log p(x, z)] + const.  "Variational inference
(ELBO max)" and "variational inference (mean-field)" are two names for
that one thing: the mean-field factorisation is the constraint, ELBO
maximisation is the objective, and neither is a separate method.

``morie.fn.vinfer`` carries the implementation and its closed-form
anchor; this module re-exports it rather than shipping a second copy.
"""

from __future__ import annotations

from .vinfer import variational_inference

__all__ = ["variational_inference"]


def cheatsheet():
    return "vbnopt: variational inference (ELBO max) -- alias of vinfer (Jordan et al. 1999)"
