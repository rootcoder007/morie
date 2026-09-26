"""plbim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plbim import plbim


def test_plbim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plbim()
