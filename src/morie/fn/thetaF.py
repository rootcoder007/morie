"""Theta method."""

__all__ = ["theta_method"]


# theta_method was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .esttsl import theta_method  # noqa: E402,F401


def cheatsheet():
    return "thetaF: Theta method"
