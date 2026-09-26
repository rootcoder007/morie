"""kgrhs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgrhs import kriging_rhs


def test_kgrhs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_rhs(values=None, x=None)
