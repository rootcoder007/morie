"""madic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.madic import madic


def test_madic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        madic()
