"""use_r2u215 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r2u215 import use_r_chapter_2_unnumbered_215


def test_use_r2u215_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_2_unnumbered_215(x=None)
