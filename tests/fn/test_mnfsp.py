"""mnfsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mnfsp import mnfsp


def test_mnfsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mnfsp()
