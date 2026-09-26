"""kgstd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgstd import kriging_std_error


def test_kgstd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_std_error(values=None, x=None)
