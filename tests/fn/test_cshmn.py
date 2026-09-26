"""cshmn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cshmn import cshmn


def test_cshmn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cshmn()
