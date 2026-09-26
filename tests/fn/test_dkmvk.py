"""dkmvk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkmvk import dkmvk


def test_dkmvk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkmvk()
