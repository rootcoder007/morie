"""use_r2u192 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r2u192 import use_r_chapter_2_unnumbered_192


def test_use_r2u192_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_2_unnumbered_192(x=None)
