"""vmmap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmmap import vmmap


def test_vmmap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmmap()
