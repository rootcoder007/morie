"""svpld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svpld import polarization_dim


def test_svpld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        polarization_dim(data=None)
