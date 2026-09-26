"""siorde is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.siorde import siorde


def test_siorde_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        siorde()
