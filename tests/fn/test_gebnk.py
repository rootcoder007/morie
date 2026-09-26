"""gebnk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gebnk import gebnk


def test_gebnk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gebnk()
