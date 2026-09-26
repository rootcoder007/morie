"""ophho is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ophho import ophho


def test_ophho_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ophho()
