"""vmpnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmpnt import vmpnt


def test_vmpnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmpnt()
