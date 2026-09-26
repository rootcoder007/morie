"""hedderich8e41 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e41 import hedderich_chapter_8_equation_41


def test_hedderich8e41_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_41(x=None)
