"""sppde is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sppde import sppde


def test_sppde_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sppde(y=None, unit_id=None)
