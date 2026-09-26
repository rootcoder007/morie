"""hedderich9u186 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u186 import hedderich_chapter_9_unnumbered_186


def test_hedderich9u186_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_186(x=None)
