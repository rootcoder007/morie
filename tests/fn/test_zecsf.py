"""zecsf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zecsf import concentration_srf


def test_zecsf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        concentration_srf(data=None)
