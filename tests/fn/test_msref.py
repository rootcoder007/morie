"""msref is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msref import reflect_config


def test_msref_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        reflect_config(data=None)
