"""tsfor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsfor import tsfor


def test_tsfor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsfor()
