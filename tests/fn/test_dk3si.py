"""dk3si is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3si import dk3si


def test_dk3si_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3si()
