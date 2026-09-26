"""svplc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svplc import plott_condition


def test_svplc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plott_condition(data=None)
