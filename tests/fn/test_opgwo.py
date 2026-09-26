"""opgwo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opgwo import opgwo


def test_opgwo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opgwo()
