"""dk3ok is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3ok import dk3ok


def test_dk3ok_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3ok()
