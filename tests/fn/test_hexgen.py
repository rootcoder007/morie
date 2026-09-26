"""hexgen is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hexgen import hexgen


def test_hexgen_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hexgen()
