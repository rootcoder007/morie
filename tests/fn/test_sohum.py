"""sohum is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sohum import sohum


def test_sohum_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sohum()
