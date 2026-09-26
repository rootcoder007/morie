"""zxmrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxmrc import mercator_proj


def test_zxmrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mercator_proj(data=None)
