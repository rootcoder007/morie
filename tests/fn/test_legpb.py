"""legpb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.legpb import legendre_basis


def test_legpb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        legendre_basis(x=None, K=None)
