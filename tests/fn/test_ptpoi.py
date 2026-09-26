"""ptpoi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptpoi import poisson_process


def test_ptpoi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        poisson_process(data=None)
