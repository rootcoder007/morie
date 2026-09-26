"""forsdi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.forsdi import forsdi


def test_forsdi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        forsdi()
