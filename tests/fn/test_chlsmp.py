"""chlsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlsmp import chlsmp


def test_chlsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlsmp()
