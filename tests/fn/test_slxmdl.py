"""slxmdl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slxmdl import slx_model


def test_slxmdl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slx_model(y=None, X=None, W=None)
