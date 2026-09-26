"""chgcus is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chgcus import changepoint_cusum


def test_chgcus_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        changepoint_cusum(y=None, k=None, h=None)
