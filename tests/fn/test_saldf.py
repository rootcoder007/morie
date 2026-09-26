"""saldf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.saldf import saldf


def test_saldf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        saldf()
