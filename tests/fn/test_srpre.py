"""srpre is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srpre import srpre


def test_srpre_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srpre()
