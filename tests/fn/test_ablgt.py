"""ablgt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ablgt import ablgt


def test_ablgt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ablgt()
