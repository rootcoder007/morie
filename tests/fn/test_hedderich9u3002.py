"""hedderich9u3002 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3002 import hedderich_chapter_9_unnumbered_3002


def test_hedderich9u3002_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3002(x=None)
