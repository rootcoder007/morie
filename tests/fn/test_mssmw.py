"""mssmw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mssmw import smacof_weight


def test_mssmw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smacof_weight(data=None)
