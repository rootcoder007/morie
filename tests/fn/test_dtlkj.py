"""dtlkj is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtlkj import dtlkj


def test_dtlkj_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtlkj()
