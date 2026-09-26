"""swstoch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swstoch import swstoch


def test_swstoch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swstoch(W=None)
