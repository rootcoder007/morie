"""agcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agcmp import agcmp


def test_agcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agcmp()
