"""use_r2u214 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r2u214 import use_r_chapter_2_unnumbered_214


def test_use_r2u214_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_2_unnumbered_214(x=None)
