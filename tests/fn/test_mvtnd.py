"""mvtnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtnd import mvtnd


def test_mvtnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtnd()
