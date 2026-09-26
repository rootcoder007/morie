"""siprob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.siprob import siprob


def test_siprob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        siprob()
