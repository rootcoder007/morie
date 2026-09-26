"""hedderich9u3533 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3533 import hedderich_chapter_9_unnumbered_3533


def test_hedderich9u3533_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3533(x=None)
