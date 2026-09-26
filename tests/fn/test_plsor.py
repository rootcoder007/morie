"""plsor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plsor import plsor


def test_plsor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plsor()
