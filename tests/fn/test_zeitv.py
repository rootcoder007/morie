"""zeitv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zeitv import travel_time_catch


def test_zeitv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        travel_time_catch(data=None)
