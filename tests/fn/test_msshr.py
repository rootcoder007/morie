"""msshr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msshr import shepard_resid


def test_msshr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        shepard_resid(data=None)
