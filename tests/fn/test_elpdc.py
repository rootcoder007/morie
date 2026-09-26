"""elpdc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elpdc import expected_log_predictive_density


def test_elpdc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        expected_log_predictive_density(log_lik=None)
