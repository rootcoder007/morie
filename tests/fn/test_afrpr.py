"""afrpr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afrpr import afrpr


def test_afrpr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afrpr()
