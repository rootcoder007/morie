"""ghcar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghcar import ghcar


def test_ghcar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghcar()
