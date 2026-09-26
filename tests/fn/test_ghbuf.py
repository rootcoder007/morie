"""ghbuf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghbuf import ghbuf


def test_ghbuf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghbuf()
