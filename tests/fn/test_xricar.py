"""xricar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xricar import icar_model


def test_xricar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        icar_model(data=None)
