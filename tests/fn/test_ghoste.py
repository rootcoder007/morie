"""ghoste is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghoste import ghose_filter


def test_ghoste_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghose_filter(smiles=None)
