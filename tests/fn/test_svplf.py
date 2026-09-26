"""svplf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svplf import polarization_aff


def test_svplf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        polarization_aff(data=None)
