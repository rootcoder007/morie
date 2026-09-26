"""mkflt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mkflt import mkflt


def test_mkflt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mkflt()
