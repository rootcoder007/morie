"""vgfml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgfml import vario_fit_ml


def test_vgfml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_fit_ml(coords=None, values=None)
