"""umsad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umsad import umsad


def test_umsad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umsad()
