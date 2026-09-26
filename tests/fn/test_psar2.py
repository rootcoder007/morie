"""psar2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.psar2 import polar_surface_area


def test_psar2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        polar_surface_area(smiles=None)
