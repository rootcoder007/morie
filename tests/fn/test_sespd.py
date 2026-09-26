"""sespd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sespd import sespd


def test_sespd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sespd()
