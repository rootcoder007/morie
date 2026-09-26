"""sagmo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sagmo import sagmo


def test_sagmo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sagmo()
