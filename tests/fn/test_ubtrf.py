"""ubtrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubtrf import ubtrf


def test_ubtrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubtrf()
