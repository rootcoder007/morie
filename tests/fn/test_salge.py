"""salge is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.salge import salge


def test_salge_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        salge()
