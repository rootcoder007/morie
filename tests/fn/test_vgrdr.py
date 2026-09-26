"""vgrdr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgrdr import rodogram


def test_vgrdr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rodogram(data=None)
