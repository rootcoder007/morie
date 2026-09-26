"""mdang is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdang import mdang


def test_mdang_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdang()
