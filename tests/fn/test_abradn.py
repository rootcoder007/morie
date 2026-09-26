"""abradn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abradn import abradn


def test_abradn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abradn()
