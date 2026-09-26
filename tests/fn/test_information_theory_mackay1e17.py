"""information_theory_mackay1e17 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.information_theory_mackay1e17 import information_theory_mackay_chapter_1_equation_17


def test_information_theory_mackay1e17_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        information_theory_mackay_chapter_1_equation_17(x=None)
