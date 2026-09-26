"""use_r2u218 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r2u218 import use_r_chapter_2_unnumbered_218


def test_use_r2u218_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_2_unnumbered_218(x=None)
