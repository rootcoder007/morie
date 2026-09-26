"""svplr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svplr import polarization_1d


def test_svplr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        polarization_1d(data=None)
