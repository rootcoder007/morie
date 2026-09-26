"""vmwht is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmwht import vmwht


def test_vmwht_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmwht()
