"""use_r2u121 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r2u121 import use_r_chapter_2_unnumbered_121


def test_use_r2u121_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_2_unnumbered_121(x=None)
