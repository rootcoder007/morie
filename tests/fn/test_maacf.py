"""maacf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maacf import maacf


def test_maacf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maacf()
