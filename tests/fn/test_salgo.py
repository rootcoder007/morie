"""salgo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.salgo import salgo


def test_salgo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        salgo()
