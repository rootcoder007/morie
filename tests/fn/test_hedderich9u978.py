"""hedderich9u978 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u978 import hedderich_chapter_9_unnumbered_978


def test_hedderich9u978_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_978(x=None)
