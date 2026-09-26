"""msnmt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msnmt import nonmetric_mds


def test_msnmt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nonmetric_mds(X=None)
