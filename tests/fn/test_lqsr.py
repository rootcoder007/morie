"""lqsr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lqsr import l1_regression


def test_lqsr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        l1_regression(X=None, y=None)
