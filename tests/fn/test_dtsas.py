"""dtsas is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtsas import dtsas


def test_dtsas_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtsas()
