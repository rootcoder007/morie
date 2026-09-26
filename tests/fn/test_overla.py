"""overla is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.overla import overla


def test_overla_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        overla()
