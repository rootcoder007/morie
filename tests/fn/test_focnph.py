"""focnph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.focnph import focnph


def test_focnph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        focnph()
