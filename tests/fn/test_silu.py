"""silu is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.silu import silu_swish


def test_silu_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        silu_swish(y=None)
