"""spccoh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spccoh import spccoh


def test_spccoh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spccoh()
