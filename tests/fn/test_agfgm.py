"""agfgm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agfgm import agfgm


def test_agfgm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agfgm()
