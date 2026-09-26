"""xrmri is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrmri import moran_resid


def test_xrmri_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        moran_resid(data=None)
