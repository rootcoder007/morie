"""bpc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bpc import bits_per_character


def test_bpc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bits_per_character(log_probs=None, N=None)
