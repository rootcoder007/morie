"""vgjck is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgjck import vario_jackknife


def test_vgjck_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_jackknife(coords=None, values=None)
