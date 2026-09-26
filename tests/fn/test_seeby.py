"""seeby is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seeby import seeby


def test_seeby_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seeby()
