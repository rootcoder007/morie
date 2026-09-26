"""svrcp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrcp import roll_call_prob


def test_svrcp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roll_call_prob(data=None)
