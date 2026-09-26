"""pains3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pains3 import pains_filter


def test_pains3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pains_filter(smiles=None)
