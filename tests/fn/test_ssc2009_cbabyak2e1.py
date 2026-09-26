"""ssc2009_cbabyak2e1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ssc2009_cbabyak2e1 import ssc2009_cbabyak_chapter_2_equation_1


def test_ssc2009_cbabyak2e1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ssc2009_cbabyak_chapter_2_equation_1(x=None)
