"""trnscl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trnscl import transitivity


def test_trnscl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        transitivity(G=None)
