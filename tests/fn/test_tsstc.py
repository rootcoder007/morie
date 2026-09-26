"""tsstc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsstc import tsstc


def test_tsstc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsstc()
