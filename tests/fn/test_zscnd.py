"""zscnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zscnd import conditional_sim


def test_zscnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        conditional_sim(data=None)
