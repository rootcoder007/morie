"""hydrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hydrf import hydrf


def test_hydrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hydrf()
