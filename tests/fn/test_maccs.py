"""maccs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maccs import maccs_keys


def test_maccs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maccs_keys(smiles=None)
