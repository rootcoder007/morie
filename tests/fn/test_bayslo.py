"""bayslo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bayslo import bayes_lasso


def test_bayslo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bayes_lasso(y=None, M=None, lam=None)
