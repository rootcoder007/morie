"""idwvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwvar import idwvar


def test_idwvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwvar()
