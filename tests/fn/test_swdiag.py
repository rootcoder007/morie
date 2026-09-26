"""swdiag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swdiag import swdiag


def test_swdiag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swdiag(W=None)
