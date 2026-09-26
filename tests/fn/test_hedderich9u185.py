"""hedderich9u185 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u185 import hedderich_chapter_9_unnumbered_185


def test_hedderich9u185_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_185(x=None)
