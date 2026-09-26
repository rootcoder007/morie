"""masst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.masst import masst


def test_masst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        masst()
