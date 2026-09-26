"""vmrob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmrob import vmrob


def test_vmrob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmrob()
