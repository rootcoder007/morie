"""zsidp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsidp import idw_power


def test_zsidp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idw_power(data=None)
