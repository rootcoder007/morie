"""grdsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.grdsmp import grdsmp


def test_grdsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        grdsmp()
