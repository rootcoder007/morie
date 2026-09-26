"""mssmc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mssmc import smacof_mds


def test_mssmc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smacof_mds(X=None)
