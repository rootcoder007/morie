"""use_r2u177 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r2u177 import use_r_chapter_2_unnumbered_177


def test_use_r2u177_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_2_unnumbered_177(x=None)
