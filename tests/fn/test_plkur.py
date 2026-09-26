"""plkur is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plkur import plkur


def test_plkur_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plkur()
