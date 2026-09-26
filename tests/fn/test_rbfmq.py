"""rbfmq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfmq import rbfmq


def test_rbfmq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfmq()
