"""ames3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ames3 import ames_mutagenicity


def test_ames3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ames_mutagenicity(smiles=None)
