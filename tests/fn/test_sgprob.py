"""sgprob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgprob import sgprob


def test_sgprob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgprob()
