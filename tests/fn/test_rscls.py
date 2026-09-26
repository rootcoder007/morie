"""rscls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rscls import rscls


def test_rscls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rscls()
