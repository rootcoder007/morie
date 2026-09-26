"""rakedw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rakedw import rake_double_weights


def test_rakedw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rake_double_weights(w_nr=None, w_cal=None)
