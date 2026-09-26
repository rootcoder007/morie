"""opcgd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opcgd import opcgd


def test_opcgd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opcgd()
