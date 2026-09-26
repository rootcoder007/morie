"""gefrm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gefrm import gefrm


def test_gefrm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gefrm()
