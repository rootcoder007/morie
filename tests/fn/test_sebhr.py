"""sebhr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sebhr import sebhr


def test_sebhr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sebhr()
