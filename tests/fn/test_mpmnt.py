"""mpmnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpmnt import mpmnt


def test_mpmnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpmnt()
