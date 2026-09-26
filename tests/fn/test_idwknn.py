"""idwknn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwknn import idwknn


def test_idwknn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwknn()
