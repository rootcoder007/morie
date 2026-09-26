"""vdwght is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vdwght import vdwght


def test_vdwght_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vdwght()
