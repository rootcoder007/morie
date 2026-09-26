"""hedderich9u3160 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3160 import hedderich_chapter_9_unnumbered_3160


def test_hedderich9u3160_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3160(x=None)
