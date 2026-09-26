"""fosimp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fosimp import fosimp


def test_fosimp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fosimp()
