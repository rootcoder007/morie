"""sebzp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sebzp import sebzp


def test_sebzp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sebzp()
