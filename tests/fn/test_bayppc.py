"""bayppc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bayppc import posterior_predictive_check


def test_bayppc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        posterior_predictive_check(y=None, y_rep=None, statistic=None)
