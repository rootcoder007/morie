"""umsto is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umsto import umsto


def test_umsto_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umsto()
