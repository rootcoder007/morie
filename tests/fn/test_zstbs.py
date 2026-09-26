"""zstbs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zstbs import turning_bands


def test_zstbs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        turning_bands(data=None)
