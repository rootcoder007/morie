"""vgfit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgfit import vario_fit_wls


def test_vgfit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_fit_wls(coords=None, values=None)
