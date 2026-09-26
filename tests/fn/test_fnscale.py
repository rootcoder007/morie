"""fnscale is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fnscale import functional_scale


def test_fnscale_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        functional_scale(f=None)
