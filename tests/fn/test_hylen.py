"""hylen is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hylen import hylen


def test_hylen_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hylen()
