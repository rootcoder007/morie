"""umcex is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umcex import umcex


def test_umcex_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umcex()
