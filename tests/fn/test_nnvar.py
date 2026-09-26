"""nnvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nnvar import nnvar


def test_nnvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nnvar()
