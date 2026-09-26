"""hypor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hypor import hypor


def test_hypor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hypor()
