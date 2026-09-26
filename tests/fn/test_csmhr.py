"""csmhr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csmhr import csmhr


def test_csmhr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csmhr()
