"""egan2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.egan2 import egan_filter


def test_egan2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        egan_filter(smiles=None)
