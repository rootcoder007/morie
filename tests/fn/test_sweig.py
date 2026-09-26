"""sweig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sweig import sweig


def test_sweig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sweig(W=None)
