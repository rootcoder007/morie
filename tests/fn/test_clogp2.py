"""clogp2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clogp2 import clogp_estimate


def test_clogp2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clogp_estimate(smiles=None)
