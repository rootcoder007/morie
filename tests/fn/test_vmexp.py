"""vmexp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmexp import vmexp


def test_vmexp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmexp()
