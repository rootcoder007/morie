"""focnp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.focnp import focnp


def test_focnp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        focnp()
