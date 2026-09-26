"""GP predictive variance"""

# gp_variance was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .gpvarF import gp_variance  # noqa: E402,F401

gp_v = gp_variance


def cheatsheet() -> str:
    return "gp_variance({}) -> GP predictive variance"
