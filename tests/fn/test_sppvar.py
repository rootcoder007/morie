"""sppvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sppvar import sppvar


def test_sppvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sppvar(resid=None, unit_id=None)
