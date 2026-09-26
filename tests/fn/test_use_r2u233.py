"""use_r2u233 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r2u233 import use_r_chapter_2_unnumbered_233


def test_use_r2u233_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_2_unnumbered_233(x=None)
