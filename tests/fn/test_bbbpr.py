"""bbbpr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bbbpr import bbb_permeability


def test_bbbpr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bbb_permeability(smiles=None)
