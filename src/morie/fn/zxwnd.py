"""Wind rose directional stats"""

# wind_rose was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .winros import wind_rose  # noqa: E402,F401

wind = wind_rose


def cheatsheet() -> str:
    return "wind_rose({}) -> Wind rose directional stats"
