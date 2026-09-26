"""sawcn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawcn import sawcn


def test_sawcn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawcn()
