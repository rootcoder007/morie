"""bayreg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bayreg import bayes_linear


def test_bayreg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bayes_linear(y=None, X=None, prior_var=None)
