"""hergp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hergp import herg_inhibition


def test_hergp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        herg_inhibition(smiles=None)
