"""tssti is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssti import tssti


def test_tssti_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssti()
