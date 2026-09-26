"""swisle is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swisle import swisle


def test_swisle_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swisle(W=None)
