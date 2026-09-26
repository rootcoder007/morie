"""selfgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.selfgr import selfies_encode


def test_selfgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        selfies_encode(smiles=None)
