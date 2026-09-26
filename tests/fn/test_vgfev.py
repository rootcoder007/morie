"""vgfev is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgfev import vario_fit_eval


def test_vgfev_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_fit_eval(coords=None, values=None)
