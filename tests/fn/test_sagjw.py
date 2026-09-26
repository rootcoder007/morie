"""sagjw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sagjw import sagjw


def test_sagjw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sagjw()
