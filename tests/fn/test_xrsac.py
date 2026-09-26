"""xrsac is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrsac import sac_ml


def test_xrsac_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sac_ml(data=None)
