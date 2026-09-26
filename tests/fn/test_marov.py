"""marov is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.marov import marov


def test_marov_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        marov()
