"""abpfc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abpfc import abpfc


def test_abpfc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abpfc()
