"""swrow is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swrow import swrow


def test_swrow_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swrow(W=None)
