"""stnsep is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stnsep import stnsep


def test_stnsep_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stnsep()
