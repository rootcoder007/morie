"""gcesm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcesm import gcesm


def test_gcesm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcesm()
