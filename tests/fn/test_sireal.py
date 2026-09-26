"""sireal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sireal import sireal


def test_sireal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sireal()
