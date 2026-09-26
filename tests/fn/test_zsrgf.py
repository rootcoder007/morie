"""zsrgf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsrgf import random_gauss_field


def test_zsrgf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        random_gauss_field(data=None)
