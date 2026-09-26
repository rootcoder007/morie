"""sgreal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgreal import sgreal


def test_sgreal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgreal()
