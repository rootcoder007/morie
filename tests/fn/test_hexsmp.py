"""hexsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hexsmp import hexsmp


def test_hexsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hexsmp()
