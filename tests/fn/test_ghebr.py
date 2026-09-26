"""ghebr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghebr import ghebr


def test_ghebr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghebr()
