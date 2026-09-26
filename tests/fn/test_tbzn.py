"""tbzn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbzn import tbzn


def test_tbzn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbzn()
