"""svrce is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrce import roll_call_error


def test_svrce_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roll_call_error(data=None)
