"""svrcs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrcs import roll_call_sim


def test_svrcs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roll_call_sim(data=None)
