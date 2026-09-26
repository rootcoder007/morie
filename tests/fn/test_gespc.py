"""gespc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gespc import gespc


def test_gespc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gespc()
