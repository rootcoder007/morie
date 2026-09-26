"""csvnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csvnd import csvnd


def test_csvnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csvnd()
