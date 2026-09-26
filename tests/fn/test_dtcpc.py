"""dtcpc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcpc import dtcpc


def test_dtcpc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcpc()
