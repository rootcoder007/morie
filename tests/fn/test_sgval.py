"""sgval is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgval import sgval


def test_sgval_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgval()
