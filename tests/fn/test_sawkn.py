"""sawkn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawkn import sawkn


def test_sawkn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawkn()
