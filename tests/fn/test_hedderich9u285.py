"""hedderich9u285 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u285 import hedderich_chapter_9_unnumbered_285


def test_hedderich9u285_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_285(x=None)
