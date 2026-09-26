"""sgann is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgann import sgann


def test_sgann_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgann()
