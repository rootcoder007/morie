"""svrps is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrps import rank_prob_score


def test_svrps_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rank_prob_score(data=None)
