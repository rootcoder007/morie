"""yllyear is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.yllyear import yll_calculation


def test_yllyear_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        yll_calculation(deaths=None, ages=None, life_table=None)
