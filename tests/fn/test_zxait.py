"""zxait is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxait import aitchison_sp


def test_zxait_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aitchison_sp(data=None)
