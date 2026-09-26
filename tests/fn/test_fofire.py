"""fofire is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fofire import fofire


def test_fofire_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fofire()
