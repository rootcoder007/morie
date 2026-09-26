"""zee2s is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zee2s import enhanced_2sfca


def test_zee2s_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enhanced_2sfca(data=None)
