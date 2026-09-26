"""vdvor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vdvor import vdvor


def test_vdvor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vdvor()
