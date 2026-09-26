"""vglik is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vglik import vario_loglik


def test_vglik_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_loglik(coords=None, values=None)
