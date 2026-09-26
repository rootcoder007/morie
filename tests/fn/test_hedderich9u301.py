"""hedderich9u301 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u301 import hedderich_chapter_9_unnumbered_301


def test_hedderich9u301_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_301(x=None)
