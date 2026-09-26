"""clopt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clopt import clopt


def test_clopt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clopt()
