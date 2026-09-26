"""reosft is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.reosft import reos_filter


def test_reosft_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        reos_filter(smiles=None)
