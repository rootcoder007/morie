"""rotbnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rotbnd import rotatable_bond_count


def test_rotbnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rotatable_bond_count(smiles=None)
