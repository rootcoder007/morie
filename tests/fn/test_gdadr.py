"""gdadr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdadr import gdadr


def test_gdadr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdadr()
