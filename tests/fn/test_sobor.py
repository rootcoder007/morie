"""sobor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sobor import sobor


def test_sobor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sobor()
