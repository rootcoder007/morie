"""plnet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plnet import plnet


def test_plnet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plnet()
