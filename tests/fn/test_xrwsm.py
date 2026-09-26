"""xrwsm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwsm import w_symmetrize


def test_xrwsm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_symmetrize(data=None)
