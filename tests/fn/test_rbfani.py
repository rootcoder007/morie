"""rbfani is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfani import rbfani


def test_rbfani_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfani()
