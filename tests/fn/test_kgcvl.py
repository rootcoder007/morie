"""kgcvl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgcvl import kriging_cv_loo


def test_kgcvl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_cv_loo(values=None, x=None)
