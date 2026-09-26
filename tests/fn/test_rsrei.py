"""rsrei is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsrei import rsrei


def test_rsrei_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsrei()
