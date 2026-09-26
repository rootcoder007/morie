"""vmani is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmani import vmani


def test_vmani_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmani()
