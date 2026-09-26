"""fodiv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fodiv import fodiv


def test_fodiv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fodiv()
