"""stsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stsim import stsim


def test_stsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stsim()
