"""mvtwe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtwe import mvtwe


def test_mvtwe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtwe()
