"""ubsho is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubsho import ubsho


def test_ubsho_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubsho()
