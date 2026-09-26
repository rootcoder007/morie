"""kglam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kglam import kriging_lambda


def test_kglam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_lambda(values=None, x=None)
