"""mtbtw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtbtw import mtbtw


def test_mtbtw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtbtw()
