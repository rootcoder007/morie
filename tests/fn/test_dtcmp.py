"""dtcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcmp import dtcmp


def test_dtcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcmp()
