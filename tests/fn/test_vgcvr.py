"""vgcvr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgcvr import covario_matrix


def test_vgcvr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        covario_matrix(coords=None, values=None)
