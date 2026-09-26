"""jrqbq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.jrqbq import jarque_bera


def test_jrqbq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        jarque_bera(resid=None)
