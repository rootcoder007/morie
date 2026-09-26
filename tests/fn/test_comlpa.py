"""comlpa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.comlpa import label_propagation


def test_comlpa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        label_propagation(G=None)
