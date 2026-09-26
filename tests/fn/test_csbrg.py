"""csbrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csbrg import csbrg


def test_csbrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csbrg()
