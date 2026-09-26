"""hedderich9u3153 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3153 import hedderich_chapter_9_unnumbered_3153


def test_hedderich9u3153_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3153(x=None)
