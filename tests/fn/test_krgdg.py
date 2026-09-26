"""krgdg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.krgdg import krgdg


def test_krgdg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        krgdg()
