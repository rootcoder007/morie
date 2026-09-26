"""gdntn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdntn import gdntn


def test_gdntn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdntn()
