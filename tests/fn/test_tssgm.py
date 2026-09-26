"""tssgm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssgm import tssgm


def test_tssgm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssgm()
