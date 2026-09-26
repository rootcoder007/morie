"""md2dp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.md2dp import md2dp


def test_md2dp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        md2dp()
