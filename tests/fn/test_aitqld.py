"""aitqld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aitqld import compositional_quantile_dist


def test_aitqld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        compositional_quantile_dist(X=None)
