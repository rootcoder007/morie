"""stngrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stngrd import stn_spatial_transform


def test_stngrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stn_spatial_transform(x=None)
