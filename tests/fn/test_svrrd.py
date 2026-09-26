"""svrrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrrd import roemer_model


def test_svrrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roemer_model(data=None)
