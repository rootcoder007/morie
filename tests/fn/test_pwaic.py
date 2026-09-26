"""pwaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pwaic import effective_parameters_waic


def test_pwaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        effective_parameters_waic(log_lik=None)
