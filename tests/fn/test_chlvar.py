"""chlvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlvar import chlvar


def test_chlvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlvar()
