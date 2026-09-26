"""hedderich9u941 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u941 import hedderich_chapter_9_unnumbered_941


def test_hedderich9u941_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_941(x=None)
