"""vmhgm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmhgm import vmhgm


def test_vmhgm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmhgm()
