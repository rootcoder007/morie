"""svdrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svdrc import svdrc


def test_svdrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svdrc()
