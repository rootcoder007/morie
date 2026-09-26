"""zsidw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsidw import idw_interp


def test_zsidw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idw_interp(data=None)
