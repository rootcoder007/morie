"""cdqnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdqnt import cdqnt


def test_cdqnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdqnt()
