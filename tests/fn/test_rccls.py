"""rccls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rccls import rccls


def test_rccls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rccls()
