"""saqmo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.saqmo import saqmo


def test_saqmo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        saqmo()
