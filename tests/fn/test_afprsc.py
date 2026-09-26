"""afprsc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afprsc import afprsc


def test_afprsc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afprsc()
