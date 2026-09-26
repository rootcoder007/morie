"""laurnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.laurnt import laurent_series


def test_laurnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        laurent_series(f=None, c=None, order=None)
