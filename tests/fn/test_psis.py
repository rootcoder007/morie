"""psis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.psis import pareto_smoothed_importance_sampling


def test_psis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pareto_smoothed_importance_sampling(log_lik=None)
