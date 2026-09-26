"""hyhlr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyhlr import hyhlr


def test_hyhlr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyhlr()
