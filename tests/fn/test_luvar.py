"""luvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.luvar import luvar


def test_luvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        luvar()
