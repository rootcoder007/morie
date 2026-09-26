"""mxtent is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mxtent import max_entropy


def test_mxtent_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        max_entropy(constraints=None)
