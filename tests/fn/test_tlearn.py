"""tlearn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tlearn import t_learner


def test_tlearn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        t_learner(y=None, D=None, X=None)
