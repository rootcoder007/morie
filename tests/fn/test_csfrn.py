"""csfrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csfrn import csfrn


def test_csfrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csfrn()
