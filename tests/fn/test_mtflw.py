"""mtflw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtflw import mtflw


def test_mtflw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtflw()
