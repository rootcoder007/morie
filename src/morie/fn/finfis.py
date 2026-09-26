"""Fisher information matrix."""

__all__ = ["fisher_information"]


# fisher_information was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .fient import fisher_information  # noqa: E402,F401


def cheatsheet():
    return "finfis: Fisher information matrix"
