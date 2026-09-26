"""zefhr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zefhr import fay_herriot


def test_zefhr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fay_herriot(data=None)
