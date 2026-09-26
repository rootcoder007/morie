"""hedderich9u940 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u940 import hedderich_chapter_9_unnumbered_940


def test_hedderich9u940_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_940(x=None)
