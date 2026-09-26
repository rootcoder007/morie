"""hepatx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hepatx import hepatotoxicity


def test_hepatx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hepatotoxicity(smiles=None)
