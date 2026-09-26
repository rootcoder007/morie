"""dk3dr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3dr import dk3dr


def test_dk3dr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3dr()
