"""baysprr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.baysprr import sparsity_horseshoe


def test_baysprr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sparsity_horseshoe(X=None, y=None, tau=None)
