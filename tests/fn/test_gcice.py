"""gcice is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcice import gcice


def test_gcice_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcice()
