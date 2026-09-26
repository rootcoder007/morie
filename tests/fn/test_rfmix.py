"""rfmix is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfmix import rfmix


def test_rfmix_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfmix()
