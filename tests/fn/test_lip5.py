"""lip5 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lip5 import lipinski_rule_of_5


def test_lip5_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lipinski_rule_of_5(smiles=None)
