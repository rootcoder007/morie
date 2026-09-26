"""svpl2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svpl2 import polarization_2d


def test_svpl2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        polarization_2d(data=None)
