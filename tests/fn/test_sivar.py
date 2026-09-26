"""sivar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sivar import sivar


def test_sivar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sivar()
