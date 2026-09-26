"""kgres is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgres import kriging_residual_map


def test_kgres_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_residual_map(values=None, x=None)
