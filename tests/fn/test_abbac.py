"""abbac is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abbac import abbac


def test_abbac_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abbac()
