"""swcomp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swcomp import swcomp


def test_swcomp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swcomp(W=None)
