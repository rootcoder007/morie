"""gdidp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdidp import gdidp


def test_gdidp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdidp()
