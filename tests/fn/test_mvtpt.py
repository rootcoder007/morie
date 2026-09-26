"""mvtpt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtpt import mvtpt


def test_mvtpt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtpt()
