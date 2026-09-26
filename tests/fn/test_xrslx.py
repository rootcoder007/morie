"""xrslx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrslx import slx_ols


def test_xrslx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slx_ols(data=None)
