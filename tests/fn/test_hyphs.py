"""hyphs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyphs import hyphs


def test_hyphs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyphs()
