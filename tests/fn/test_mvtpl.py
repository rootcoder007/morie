"""mvtpl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtpl import mvtpl


def test_mvtpl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtpl()
