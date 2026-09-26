"""hedderich9u3104 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3104 import hedderich_chapter_9_unnumbered_3104


def test_hedderich9u3104_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3104(x=None)
