"""bayocl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bayocl import bayes_outlier_dp


def test_bayocl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bayes_outlier_dp(y=None, alpha=None)
