"""use_r2u189 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r2u189 import use_r_chapter_2_unnumbered_189


def test_use_r2u189_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_2_unnumbered_189(x=None)
