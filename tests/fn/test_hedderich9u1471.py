"""hedderich9u1471 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1471 import hedderich_chapter_9_unnumbered_1471


def test_hedderich9u1471_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1471(x=None)
