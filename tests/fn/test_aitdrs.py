"""aitdrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aitdrs import dirichlet_sample


def test_aitdrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dirichlet_sample(alpha=None, n=None)
