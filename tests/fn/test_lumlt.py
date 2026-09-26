"""lumlt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lumlt import lumlt


def test_lumlt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lumlt()
