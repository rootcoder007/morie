"""hedderich9u2852 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u2852 import hedderich_chapter_9_unnumbered_2852


def test_hedderich9u2852_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_2852(x=None)
