"""gdred is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdred import gdred


def test_gdred_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdred()
