"""svdir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svdir import svdir


def test_svdir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svdir()
