"""kgcvk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgcvk import kriging_cv_kfold


def test_kgcvk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_cv_kfold(values=None, x=None)
