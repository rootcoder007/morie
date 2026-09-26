"""hedderich9u1510 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1510 import hedderich_chapter_9_unnumbered_1510


def test_hedderich9u1510_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1510(x=None)
