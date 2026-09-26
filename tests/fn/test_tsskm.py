"""tsskm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsskm import tsskm


def test_tsskm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsskm()
