"""sbvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbvar import sbvar


def test_sbvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbvar()
