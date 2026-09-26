"""nnwgt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nnwgt import nnwgt


def test_nnwgt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nnwgt()
