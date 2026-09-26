"""vmfit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmfit import vmfit


def test_vmfit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmfit()
