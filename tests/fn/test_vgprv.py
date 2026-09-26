"""vgprv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgprv import pairwise_rel_vario


def test_vgprv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pairwise_rel_vario(coords=None, values=None)
