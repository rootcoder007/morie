"""splfn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.splfn import splfn


def test_splfn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        splfn()
