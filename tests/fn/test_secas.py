"""secas is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.secas import secas


def test_secas_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        secas()
