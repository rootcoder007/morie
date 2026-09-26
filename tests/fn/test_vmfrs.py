"""vmfrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmfrs import vmfrs


def test_vmfrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmfrs()
