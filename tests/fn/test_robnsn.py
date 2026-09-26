"""robnsn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.robnsn import robnsn


def test_robnsn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        robnsn()
