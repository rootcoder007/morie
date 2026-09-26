"""gpdeep is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpdeep import gpdeep


def test_gpdeep_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpdeep()
