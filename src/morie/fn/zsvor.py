"""Voronoi polygon areas"""

# voronoi_areas was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .voron import voronoi_areas  # noqa: E402,F401

voro = voronoi_areas


def cheatsheet() -> str:
    return "voronoi_areas({}) -> Voronoi polygon areas"
