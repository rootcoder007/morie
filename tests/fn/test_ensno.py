"""ensno is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ensno import ensno


def test_ensno_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ensno()
