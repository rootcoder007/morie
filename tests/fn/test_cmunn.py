"""cmunn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmunn import cmunn


def test_cmunn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmunn()
