"""svple is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svple import polarization_er


def test_svple_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        polarization_er(data=None)
