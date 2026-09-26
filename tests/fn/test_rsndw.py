"""rsndw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsndw import rsndw


def test_rsndw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsndw()
