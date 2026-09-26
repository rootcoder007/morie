"""kgunv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgunv import uk_variance


def test_kgunv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        uk_variance(data=None)
