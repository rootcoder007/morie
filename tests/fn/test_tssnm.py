"""tssnm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssnm import tssnm


def test_tssnm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssnm()
