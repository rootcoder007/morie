"""krgre is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.krgre import krgre


def test_krgre_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        krgre()
