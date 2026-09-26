"""bayoutl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bayoutl import bayes_outlier


def test_bayoutl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bayes_outlier(y=None, outlier_prior=None)
