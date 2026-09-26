"""agfdi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agfdi import agfdi


def test_agfdi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agfdi()
