"""thetaF is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.thetaF import theta_method


def test_thetaF_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        theta_method(y=None, theta=None)
