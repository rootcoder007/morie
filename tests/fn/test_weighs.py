"""weighs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.weighs import weighs


def test_weighs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        weighs()
