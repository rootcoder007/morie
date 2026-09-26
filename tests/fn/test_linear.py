"""linear is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.linear import linearization_se


def test_linear_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        linearization_se(estimator=None, data=None)
