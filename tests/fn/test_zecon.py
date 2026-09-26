"""zecon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zecon import concentration_idx


def test_zecon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        concentration_idx(data=None)
