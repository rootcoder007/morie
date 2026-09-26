"""plvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plvar import plvar


def test_plvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plvar()
