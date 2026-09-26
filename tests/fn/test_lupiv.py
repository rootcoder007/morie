"""lupiv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lupiv import lupiv


def test_lupiv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lupiv()
