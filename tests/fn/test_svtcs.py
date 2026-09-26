"""svtcs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svtcs import top_cycle_set


def test_svtcs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        top_cycle_set(data=None)
