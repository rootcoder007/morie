"""hygrw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hygrw import hygrw


def test_hygrw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hygrw()
