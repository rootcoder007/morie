"""sfeig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sfeig import sfeig


def test_sfeig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sfeig(y=None, X=None, W=None)
