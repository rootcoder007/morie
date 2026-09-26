"""hedderich9u3133 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3133 import hedderich_chapter_9_unnumbered_3133


def test_hedderich9u3133_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3133(x=None)
