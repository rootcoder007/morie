"""svhtd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svhtd import hotelling_downs


def test_svhtd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hotelling_downs(data=None)
