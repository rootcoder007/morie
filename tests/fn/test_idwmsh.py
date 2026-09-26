"""idwmsh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwmsh import idwmsh


def test_idwmsh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwmsh()
