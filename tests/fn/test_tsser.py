"""tsser is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsser import tsser


def test_tsser_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsser()
