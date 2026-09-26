"""xrgwc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgwc import gwr_coefficients


def test_xrgwc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwr_coefficients(data=None)
