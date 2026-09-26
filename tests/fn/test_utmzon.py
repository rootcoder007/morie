"""utmzon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.utmzon import utmzon


def test_utmzon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        utmzon()
