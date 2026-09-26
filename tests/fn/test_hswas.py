"""hswas is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hswas import hswas


def test_hswas_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hswas()
