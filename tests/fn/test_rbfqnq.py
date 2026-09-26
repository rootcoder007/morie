"""rbfqnq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfqnq import rbfqnq


def test_rbfqnq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfqnq()
