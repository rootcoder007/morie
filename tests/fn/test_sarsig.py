"""sarsig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sarsig import sarsig


def test_sarsig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sarsig(resid=None, n=None)
