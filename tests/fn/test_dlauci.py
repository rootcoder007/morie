"""dlauci is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dlauci import dlauci


def test_dlauci_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dlauci()
