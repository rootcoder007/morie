"""umkkt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umkkt import umkkt


def test_umkkt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umkkt()
