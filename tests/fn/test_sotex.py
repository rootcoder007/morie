"""sotex is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sotex import sotex


def test_sotex_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sotex()
