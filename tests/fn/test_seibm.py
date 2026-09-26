"""seibm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seibm import seibm


def test_seibm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seibm()
