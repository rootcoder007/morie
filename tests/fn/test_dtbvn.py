"""dtbvn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtbvn import dtbvn


def test_dtbvn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtbvn()
