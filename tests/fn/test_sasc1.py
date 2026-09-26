"""sasc1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sasc1 import synthetic_accessibility


def test_sasc1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        synthetic_accessibility(smiles=None)
