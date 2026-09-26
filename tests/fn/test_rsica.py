"""rsica is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsica import rsica


def test_rsica_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsica()
