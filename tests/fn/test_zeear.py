"""zeear is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zeear import ecological_reg


def test_zeear_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ecological_reg(data=None)
