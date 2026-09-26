"""opsca is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opsca import opsca


def test_opsca_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opsca()
