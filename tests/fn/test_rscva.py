"""rscva is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rscva import rscva


def test_rscva_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rscva()
