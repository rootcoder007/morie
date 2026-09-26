"""hedderich9u305 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u305 import hedderich_chapter_9_unnumbered_305


def test_hedderich9u305_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_305(x=None)
