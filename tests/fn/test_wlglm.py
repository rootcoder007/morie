"""wlglm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlglm import wlglm


def test_wlglm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlglm()
