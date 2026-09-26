"""Join count statistic"""

# join_count was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .jjmsta import join_count  # noqa: E402,F401

join = join_count


def cheatsheet() -> str:
    return "join_count({}) -> Join count statistic"
