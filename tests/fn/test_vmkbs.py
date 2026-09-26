"""vmkbs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmkbs import vmkbs


def test_vmkbs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmkbs()
