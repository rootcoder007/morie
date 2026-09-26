"""zxlam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxlam import lambert_proj


def test_zxlam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lambert_proj(data=None)
