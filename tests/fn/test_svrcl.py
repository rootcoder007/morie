"""svrcl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrcl import roll_call_logit


def test_svrcl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roll_call_logit(data=None)
