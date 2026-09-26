"""idpdw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpdw import idpdw


def test_idpdw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpdw()
