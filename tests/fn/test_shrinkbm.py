"""shrinkbm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.shrinkbm import shrinkage_bayes


def test_shrinkbm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        shrinkage_bayes(X=None, y=None, prior_family=None)
