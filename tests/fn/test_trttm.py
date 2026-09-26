"""trttm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trttm import trttm


def test_trttm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trttm()
