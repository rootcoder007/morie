"""umlag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umlag import umlag


def test_umlag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umlag()
