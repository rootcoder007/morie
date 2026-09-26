"""naoIdx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.naoIdx import nao_index


def test_naoIdx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nao_index(slp=None)
