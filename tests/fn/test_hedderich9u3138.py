"""hedderich9u3138 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3138 import hedderich_chapter_9_unnumbered_3138


def test_hedderich9u3138_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3138(x=None)
