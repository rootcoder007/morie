"""Jarque-Bera normality on residuals."""

__all__ = ["jarque_bera"]


# jarque_bera was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .jarber import jarque_bera  # noqa: E402,F401


def cheatsheet():
    return "jrqbq: Jarque-Bera normality on residuals"
