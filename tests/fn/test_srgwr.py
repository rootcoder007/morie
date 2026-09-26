"""srgwr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srgwr import srgwr


def test_srgwr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srgwr()
