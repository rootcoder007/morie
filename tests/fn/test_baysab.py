"""baysab is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.baysab import bayes_a_alpha


def test_baysab_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bayes_a_alpha(y=None, M=None)
