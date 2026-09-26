"""rsnds is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsnds import rsnds


def test_rsnds_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsnds()
