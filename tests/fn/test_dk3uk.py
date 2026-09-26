"""dk3uk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3uk import dk3uk


def test_dk3uk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3uk()
