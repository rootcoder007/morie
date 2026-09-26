"""rscir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rscir import rscir


def test_rscir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rscir()
