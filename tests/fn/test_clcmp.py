"""clcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clcmp import clcmp


def test_clcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clcmp()
