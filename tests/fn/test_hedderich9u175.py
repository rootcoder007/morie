"""hedderich9u175 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u175 import hedderich_chapter_9_unnumbered_175


def test_hedderich9u175_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_175(x=None)
