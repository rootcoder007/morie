"""stdea is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stdea import st_diff_equation


def test_stdea_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        st_diff_equation()
