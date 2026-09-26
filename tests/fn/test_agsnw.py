"""agsnw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agsnw import agsnw


def test_agsnw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agsnw()
