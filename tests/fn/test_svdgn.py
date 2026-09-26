"""svdgn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svdgn import deegan_packel


def test_svdgn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        deegan_packel(data=None)
