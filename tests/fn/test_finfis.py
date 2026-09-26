"""finfis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.finfis import fisher_information


def test_finfis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fisher_information(log_likelihood=None, theta=None)
