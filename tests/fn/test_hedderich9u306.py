"""hedderich9u306 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u306 import hedderich_chapter_9_unnumbered_306


def test_hedderich9u306_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_306(x=None)
