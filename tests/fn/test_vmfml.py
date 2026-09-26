"""vmfml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmfml import vmfml


def test_vmfml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmfml()
