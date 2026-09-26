"""rbfimq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfimq import rbfimq


def test_rbfimq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfimq()
