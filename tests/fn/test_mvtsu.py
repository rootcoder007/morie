"""mvtsu is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtsu import mvtsu


def test_mvtsu_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtsu()
