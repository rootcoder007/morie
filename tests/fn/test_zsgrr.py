"""zsgrr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsgrr import grid_resample


def test_zsgrr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        grid_resample(data=None)
