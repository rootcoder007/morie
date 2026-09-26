"""afchl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afchl import afchl


def test_afchl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afchl()
