"""svwht is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svwht import wittman_model


def test_svwht_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wittman_model(data=None)
