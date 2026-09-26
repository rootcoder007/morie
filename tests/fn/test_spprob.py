"""spprob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spprob import spprob


def test_spprob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spprob(y=None, X=None, W=None)
