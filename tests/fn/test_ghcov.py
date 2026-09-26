"""ghcov is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghcov import ghcov


def test_ghcov_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghcov()
