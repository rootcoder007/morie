"""mcint is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcint import mcint


def test_mcint_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcint()
