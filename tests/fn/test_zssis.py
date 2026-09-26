"""zssis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zssis import seq_ind_sim


def test_zssis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seq_ind_sim(data=None)
