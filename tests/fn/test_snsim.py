"""snsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.snsim import snsim


def test_snsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        snsim()
