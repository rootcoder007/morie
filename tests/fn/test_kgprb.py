"""kgprb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgprb import kriging_prob_map


def test_kgprb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_prob_map(values=None, x=None)
