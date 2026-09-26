"""hedderich9u2971 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u2971 import hedderich_chapter_9_unnumbered_2971


def test_hedderich9u2971_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_2971(x=None)
