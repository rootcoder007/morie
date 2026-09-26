"""rhatc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rhatc import r_hat_convergence


def test_rhatc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        r_hat_convergence(chains=None)
