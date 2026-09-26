"""rsmnf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsmnf import rsmnf


def test_rsmnf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsmnf()
