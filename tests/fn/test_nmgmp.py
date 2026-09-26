"""nmgmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmgmp import gmp_stat


def test_nmgmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gmp_stat(data=None)
