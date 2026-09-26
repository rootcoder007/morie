"""fovol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fovol import fovol


def test_fovol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fovol()
