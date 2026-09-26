"""mwght is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mwght import molecular_weight


def test_mwght_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        molecular_weight(smiles=None)
