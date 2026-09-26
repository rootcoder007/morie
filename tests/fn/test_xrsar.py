"""xrsar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrsar import sar_ml


def test_xrsar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sar_ml(data=None)
