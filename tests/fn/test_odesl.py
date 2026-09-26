"""odesl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.odesl import ode_symbolic


def test_odesl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ode_symbolic(ode=None)
