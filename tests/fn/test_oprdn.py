"""oprdn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.oprdn import oprdn


def test_oprdn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oprdn()
