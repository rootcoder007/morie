"""zelrx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zelrx import leroux_model


def test_zelrx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        leroux_model(data=None)
