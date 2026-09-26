"""hedderich9u187 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u187 import hedderich_chapter_9_unnumbered_187


def test_hedderich9u187_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_187(x=None)
