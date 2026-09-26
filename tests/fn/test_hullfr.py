"""hullfr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hullfr import hullfr


def test_hullfr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hullfr()
