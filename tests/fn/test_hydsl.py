"""hydsl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hydsl import hydsl


def test_hydsl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hydsl()
