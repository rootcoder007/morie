"""kgrms is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgrms import kriging_rmse


def test_kgrms_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_rmse(values=None, x=None)
