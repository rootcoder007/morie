"""svrel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrel import svrel


def test_svrel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svrel()
