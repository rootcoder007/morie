"""tqpol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tqpol import turboquant_polar_transform


def test_tqpol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        turboquant_polar_transform(x=None)
