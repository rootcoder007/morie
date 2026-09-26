"""tsste is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsste import tsste


def test_tsste_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsste()
