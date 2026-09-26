"""tsstd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsstd import tsstd


def test_tsstd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsstd()
