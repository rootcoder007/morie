"""pollR is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pollR import pollards_rho


def test_pollR_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pollards_rho(n=None)
