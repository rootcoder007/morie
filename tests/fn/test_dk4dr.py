"""dk4dr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk4dr import dk4dr


def test_dk4dr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk4dr()
