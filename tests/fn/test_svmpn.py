"""svmpn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svmpn import multiparty_nash


def test_svmpn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        multiparty_nash(data=None)
