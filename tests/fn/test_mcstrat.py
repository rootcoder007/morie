"""mcstrat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcstrat import mcstrat


def test_mcstrat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcstrat()
