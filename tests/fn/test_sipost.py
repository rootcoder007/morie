"""sipost is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sipost import sipost


def test_sipost_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sipost()
