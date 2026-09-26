"""vdrefn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vdrefn import vdrefn


def test_vdrefn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vdrefn()
