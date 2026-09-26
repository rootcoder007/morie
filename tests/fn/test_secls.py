"""secls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.secls import secls


def test_secls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        secls()
