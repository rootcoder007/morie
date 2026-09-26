"""trmod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trmod import trmod


def test_trmod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trmod()
