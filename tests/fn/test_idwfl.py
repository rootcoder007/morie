"""idwfl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwfl import idwfl


def test_idwfl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwfl()
