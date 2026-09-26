"""nnlap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nnlap import nnlap


def test_nnlap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nnlap()
