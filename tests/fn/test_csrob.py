"""csrob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csrob import csrob


def test_csrob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csrob()
