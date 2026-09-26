"""smspln is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.smspln import smoothing_spline


def test_smspln_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smoothing_spline(x=None, y=None, lam=None)
