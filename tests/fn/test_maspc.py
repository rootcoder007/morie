"""maspc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maspc import maspc


def test_maspc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maspc()
