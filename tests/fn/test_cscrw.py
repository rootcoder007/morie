"""cscrw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cscrw import cscrw


def test_cscrw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cscrw()
