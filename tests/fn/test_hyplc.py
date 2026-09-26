"""hyplc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyplc import harmonic_mean_estimator


def test_hyplc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        harmonic_mean_estimator(log_lik=None)
