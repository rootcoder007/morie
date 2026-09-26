"""rcabs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcabs import rcabs


def test_rcabs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcabs()
