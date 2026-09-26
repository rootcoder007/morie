"""galois is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.galois import galois_group


def test_galois_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        galois_group(poly=None)
