"""forecr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.forecr import forecr


def test_forecr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        forecr()
