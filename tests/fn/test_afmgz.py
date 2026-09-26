"""afmgz is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afmgz import afmgz


def test_afmgz_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afmgz()
