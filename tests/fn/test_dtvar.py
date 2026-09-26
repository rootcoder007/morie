"""dtvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtvar import dtvar


def test_dtvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtvar()
