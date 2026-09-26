"""zstrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zstrd import trend_temporal


def test_zstrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trend_temporal(data=None)
