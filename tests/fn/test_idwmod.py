"""idwmod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwmod import idwmod


def test_idwmod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwmod()
