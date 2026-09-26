"""spprocc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spprocc import spprocc


def test_spprocc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spprocc(y=None, probs=None)
