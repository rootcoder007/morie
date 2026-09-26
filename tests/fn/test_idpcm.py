"""idpcm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpcm import idpcm


def test_idpcm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpcm()
