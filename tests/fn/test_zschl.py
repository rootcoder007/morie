"""zschl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zschl import chol_sim


def test_zschl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chol_sim(data=None)
