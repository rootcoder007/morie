"""tsvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsvar import tsvar


def test_tsvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsvar()
