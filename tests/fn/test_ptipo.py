"""ptipo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptipo import inhom_poisson


def test_ptipo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        inhom_poisson(data=None)
