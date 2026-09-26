"""dtaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtaic import dtaic


def test_dtaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtaic()
