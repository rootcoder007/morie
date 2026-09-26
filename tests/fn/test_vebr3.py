"""vebr3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vebr3 import veber_rule


def test_vebr3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        veber_rule(smiles=None)
