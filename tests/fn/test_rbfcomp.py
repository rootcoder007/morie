"""rbfcomp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfcomp import rbfcomp


def test_rbfcomp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfcomp()
