"""tssae is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssae import tssae


def test_tssae_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssae()
