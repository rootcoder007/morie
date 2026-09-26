"""aitdrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aitdrf import dirichlet_fit_mom


def test_aitdrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dirichlet_fit_mom(X=None)
