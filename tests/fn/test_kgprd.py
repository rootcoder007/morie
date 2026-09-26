"""kgprd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgprd import kriging_predict


def test_kgprd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_predict(values=None, x=None)
