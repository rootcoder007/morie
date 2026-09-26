"""vmsph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmsph import vmsph


def test_vmsph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmsph()
