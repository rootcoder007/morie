"""zeism is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zeism import indirect_std


def test_zeism_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        indirect_std(data=None)
